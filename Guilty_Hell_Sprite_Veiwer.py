import pygame
import pygame.freetype
from PIL import Image
from pathlib import Path
from collections import defaultdict
import re
import sys
import threading
import queue
import os
import urllib.request

try:
    import numpy as np
except ImportError:
    np = None
    print("NumPy not available - GPU upscaling disabled")

# Try to import AI upscaling dependencies
UPSCALE_AVAILABLE = False
CUDA_DEVICE = None
TORCH_VERSION = None
UPSCALE_METHOD = None

try:
    import torch
    TORCH_VERSION = torch.__version__
    print(f"PyTorch version: {TORCH_VERSION}")
    print(f"CUDA built: {torch.version.cuda if torch.version.cuda else 'NO - CPU ONLY BUILD'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        CUDA_DEVICE = torch.cuda.get_device_name(0)
        print(f"GPU: {CUDA_DEVICE}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Try Real-ESRGAN first
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            UPSCALE_AVAILABLE = True
            UPSCALE_METHOD = "RealESRGAN"
            print("Real-ESRGAN loaded successfully!")
        except ImportError as e:
            print(f"Real-ESRGAN not available: {e}")
            
            # Try using torch.hub to load ESRGAN directly
            try:
                # Alternative: use a simpler upscaling with torch
                print("Trying alternative CUDA upscaling...")
                UPSCALE_AVAILABLE = True
                UPSCALE_METHOD = "CUDA_LANCZOS"
                print("CUDA Lanczos upscaling enabled")
            except Exception as e2:
                print(f"Alternative also failed: {e2}")
    else:
        print("\n" + "="*60)
        print("WARNING: CUDA NOT AVAILABLE!")
        print("You have an RTX 4090 but PyTorch can't use it.")
        print("\nTo fix, run these commands:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("="*60 + "\n")
except ImportError as e:
    print(f"PyTorch not available: {e}")
    print("\nTo install AI upscaling:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("  pip install realesrgan basicsr")

# Pattern: Name_Animation_W_H_Gap_OffX_OffY.png (optional #number suffix)
SPRITE_PATTERN = re.compile(r'^(.+?)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)(?:\s*#\d+)?\.png$')

MODEL_URLS = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
}

def download_model(model_name, dest_folder=None):
    """Download model if not present"""
    if dest_folder is None:
        dest_folder = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(dest_folder, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    if model_name not in MODEL_URLS:
        print(f"Unknown model: {model_name}")
        return None
    
    url = MODEL_URLS[model_name]
    print(f"Downloading {model_name} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, model_path, 
            lambda count, block, total: print(f"\rDownloading: {count * block / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB", end=''))
        print(f"\nSaved to {model_path}")
        return model_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def scan_directory(directory):
    """Scan directory and build character -> animation -> files tree"""
    tree = defaultdict(lambda: defaultdict(list))
    
    for f in Path(directory).glob('*.png'):
        name = f.name
        match = SPRITE_PATTERN.search(name)
        if match:
            prefix = match.group(1)
            parts = prefix.rsplit('_', 1)
            if len(parts) == 2:
                char, anim = parts
            else:
                char = parts[0]
                anim = "Default"
            
            if '#' in name:
                base_name = re.sub(r'\s*#\d+', '', name)
                base_path = f.parent / base_name
                if base_path.exists():
                    continue
            
            tree[char][anim].append(f)
    
    sorted_tree = {}
    for char in sorted(tree.keys()):
        sorted_tree[char] = {}
        for anim in sorted(tree[char].keys()):
            sorted_tree[char][anim] = sorted(tree[char][anim])
    
    return sorted_tree


def parse_sprite_sheet(filepath):
    """Parse sprite sheet parameters from filename"""
    match = SPRITE_PATTERN.search(filepath.name)
    if not match:
        return None
    
    prefix = match.group(1)
    w, h, gap = int(match.group(2)), int(match.group(3)), int(match.group(4))
    offset = (int(match.group(5)), int(match.group(6)))
    
    parts = prefix.rsplit('_', 1)
    char = parts[0] if len(parts) == 2 else prefix
    anim = parts[1] if len(parts) == 2 else "Default"
    
    return {
        'char': char,
        'anim': anim,
        'w': w,
        'h': h,
        'gap': gap,
        'offset': offset,
        'path': filepath
    }


def extract_frames(filepath, alpha_threshold=0):
    """Extract non-empty frames from sprite sheet"""
    info = parse_sprite_sheet(filepath)
    if not info:
        return [], None
    
    try:
        img = Image.open(filepath).convert('RGBA')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return [], info
    
    sheet_w, sheet_h = img.size
    w, h, gap = info['w'], info['h'], info['gap']
    
    cols = (sheet_w + gap) // (w + gap) if (w + gap) > 0 else 1
    rows = (sheet_h + gap) // (h + gap) if (h + gap) > 0 else 1
    
    frames = []
    for row in range(rows):
        for col in range(cols):
            x = col * (w + gap)
            y = row * (h + gap)
            
            if x + w > sheet_w or y + h > sheet_h:
                continue
                
            tile = img.crop((x, y, x + w, y + h))
            alpha = tile.getchannel('A')
            if alpha.getextrema()[1] > alpha_threshold:
                frames.append(tile)
    
    return frames, info


def pil_to_pygame(pil_img):
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert_alpha()


class PILUpscaler:
    """Fallback upscaler using PIL's high-quality Lanczos resampling with sharpening"""
    
    def __init__(self, scale=8):
        self.scale = scale
        self.ready = True
        self.loading = False
        self.status_message = "PIL Lanczos (CPU)"
        self.sharpen_amount = 1.5  # PIL needs a bit more
    
    def load_model(self):
        return True
    
    def upscale_pil(self, pil_img):
        """Upscale using PIL's LANCZOS with sharpening"""
        try:
            from PIL import ImageFilter, ImageEnhance
            
            new_size = (pil_img.width * self.scale, pil_img.height * self.scale)
            result = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply unsharp mask for sharpening
            if self.sharpen_amount > 0:
                # UnsharpMask(radius, percent, threshold)
                result = result.filter(ImageFilter.UnsharpMask(
                    radius=2, 
                    percent=int(100 * self.sharpen_amount), 
                    threshold=1
                ))
            
            return result
        except Exception as e:
            print(f"PIL upscale error: {e}")
            return None


class CUDAUpscaler:
    """GPU-accelerated upscaler using PyTorch with sharpening"""
    
    def __init__(self, scale=8):
        self.scale = scale
        self.ready = False
        self.loading = False
        self.status_message = "Initializing CUDA..."
        self.device = None
        self.sharpen_amount = 1.0  # 0 = off, 1 = normal, 2 = strong
    
    def load_model(self):
        try:
            self.loading = True
            self.status_message = "Setting up CUDA..."
            import torch
            self.device = torch.device('cuda')
            # Warm up CUDA
            test = torch.zeros(1, device=self.device)
            del test
            self.ready = True
            self.loading = False
            self.status_message = "CUDA Upscale Ready"
            print(f"CUDA upscaler ready! Scale: {self.scale}x, Sharpen: {self.sharpen_amount}")
            return True
        except Exception as e:
            self.loading = False
            self.status_message = f"CUDA failed: {str(e)[:20]}"
            print(f"CUDA upscaler failed: {e}")
            return False
    
    def sharpen_tensor(self, img_tensor):
        """Apply unsharp mask sharpening on GPU"""
        import torch
        import torch.nn.functional as F
        
        if self.sharpen_amount <= 0:
            return img_tensor
        
        # Create Gaussian blur kernel for unsharp mask
        kernel_size = 5
        sigma = 1.0
        
        # Generate 1D Gaussian kernel
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D kernel
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        
        # Apply to each channel
        channels = img_tensor.shape[1]
        gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
        
        # Pad and apply blur
        pad = kernel_size // 2
        padded = F.pad(img_tensor, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(padded, gauss_2d, groups=channels)
        
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = img_tensor + self.sharpen_amount * (img_tensor - blurred)
        
        return sharpened.clamp(0, 1)
    
    def upscale_pil(self, pil_img):
        """Upscale using PyTorch on GPU with bicubic interpolation + sharpening"""
        if not self.ready:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Handle RGBA
            if pil_img.mode == 'RGBA':
                rgb = pil_img.convert('RGB')
                alpha = pil_img.getchannel('A')
            else:
                rgb = pil_img
                alpha = None
            
            # Convert to tensor [1, C, H, W]
            img_np = np.array(rgb).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Calculate new size
            new_h = pil_img.height * self.scale
            new_w = pil_img.width * self.scale
            
            # Upscale using bicubic interpolation on GPU
            upscaled = F.interpolate(img_tensor, size=(new_h, new_w), 
                                     mode='bicubic', align_corners=False)
            upscaled = upscaled.clamp(0, 1)
            
            # Apply sharpening
            sharpened = self.sharpen_tensor(upscaled)
            
            # Convert back
            result_np = (sharpened.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            result = Image.fromarray(result_np)
            
            # Handle alpha channel
            if alpha is not None:
                alpha_up = alpha.resize((new_w, new_h), Image.Resampling.LANCZOS)
                result = result.convert('RGBA')
                result.putalpha(alpha_up)
            
            return result
            
        except Exception as e:
            print(f"CUDA upscale error: {e}")
            import traceback
            traceback.print_exc()
            return None


class AIUpscaler:
    """Real-ESRGAN AI upscaler with CUDA acceleration and sharpening"""
    
    def __init__(self, scale=4):
        self.scale = scale  # Real-ESRGAN max is 4x
        self.model = None
        self.upscaler = None
        self.ready = False
        self.loading = False
        self.load_error = None
        self.status_message = "Initializing..."
        self.sharpen_amount = 0.5  # Less needed for AI upscale
        
    def load_model(self):
        """Load the Real-ESRGAN model"""
        if not UPSCALE_AVAILABLE:
            self.load_error = "CUDA/Real-ESRGAN not available"
            self.status_message = "AI not available"
            return False
            
        try:
            self.loading = True
            self.status_message = "Downloading model..."
            
            # Download model if needed
            model_name = f'RealESRGAN_x{self.scale}plus.pth'
            model_path = download_model(model_name)
            
            if not model_path or not os.path.exists(model_path):
                self.load_error = f"Could not download {model_name}"
                self.status_message = "Model download failed"
                self.loading = False
                return False
            
            self.status_message = "Loading AI model..."
            print(f"Loading Real-ESRGAN model from {model_path}...")
            
            # Create model architecture
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=self.scale
            )
            
            self.upscaler = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                dni_weight=None,
                model=model,
                tile=0,  # No tiling - use full GPU
                tile_pad=10,
                pre_pad=0,
                half=True,  # FP16 for speed on RTX
                gpu_id=0
            )
            
            self.ready = True
            self.loading = False
            self.status_message = "AI Ready"
            print("Real-ESRGAN model loaded successfully!")
            return True
            
        except Exception as e:
            self.load_error = str(e)
            self.loading = False
            self.status_message = f"Load failed: {str(e)[:30]}"
            print(f"Failed to load Real-ESRGAN: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upscale_pil(self, pil_img):
        """Upscale a PIL image and return PIL image"""
        if not self.ready:
            return None
            
        try:
            from PIL import ImageFilter
            
            # Handle RGBA
            if pil_img.mode == 'RGBA':
                rgb = pil_img.convert('RGB')
                alpha = pil_img.getchannel('A')
            else:
                rgb = pil_img
                alpha = None
            
            # Convert to numpy BGR for Real-ESRGAN
            img_np = np.array(rgb)[:, :, ::-1].copy()
            
            # Upscale
            output, _ = self.upscaler.enhance(img_np, outscale=self.scale)
            
            # Convert back to RGB PIL
            output_rgb = output[:, :, ::-1]
            result = Image.fromarray(output_rgb)
            
            # Apply light sharpening
            if self.sharpen_amount > 0:
                result = result.filter(ImageFilter.UnsharpMask(
                    radius=1.5, 
                    percent=int(80 * self.sharpen_amount), 
                    threshold=1
                ))
            
            # Upscale and recombine alpha
            if alpha is not None:
                alpha_up = alpha.resize(result.size, Image.Resampling.LANCZOS)
                result = result.convert('RGBA')
                result.putalpha(alpha_up)
            
            return result
            
        except Exception as e:
            print(f"Upscale error: {e}")
            import traceback
            traceback.print_exc()
            return None


class Camera:
    """Handles pan and zoom for the preview area"""
    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 16.0
    
    def reset(self):
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
    
    def pan(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
    
    def zoom_at(self, factor, screen_x, screen_y, center_x, center_y):
        old_zoom = self.zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        
        if old_zoom != self.zoom:
            world_x = (screen_x - center_x - self.offset_x) / old_zoom
            world_y = (screen_y - center_y - self.offset_y) / old_zoom
            new_screen_x = world_x * self.zoom + center_x + self.offset_x
            new_screen_y = world_y * self.zoom + center_y + self.offset_y
            self.offset_x -= (new_screen_x - screen_x)
            self.offset_y -= (new_screen_y - screen_y)
    
    def world_to_screen(self, wx, wy, center_x, center_y):
        sx = wx * self.zoom + center_x + self.offset_x
        sy = wy * self.zoom + center_y + self.offset_y
        return sx, sy
    
    def screen_to_world(self, sx, sy, center_x, center_y):
        wx = (sx - center_x - self.offset_x) / self.zoom
        wy = (sy - center_y - self.offset_y) / self.zoom
        return wx, wy


class SpriteViewer:
    def __init__(self, directory, preselect_file=None):
        pygame.init()
        
        self.screen_w, self.screen_h = 1400, 900
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
        pygame.display.set_caption("Sprite Animation Viewer")
        
        self.font = pygame.freetype.SysFont('Consolas', 14)
        self.font_small = pygame.freetype.SysFont('Consolas', 12)
        
        self.tree_width = 350
        self.scroll_y = 0
        self.scroll_speed = 20
        self.max_scroll = 0
        
        # Colors
        self.bg_color = (30, 30, 35)
        self.tree_bg = (40, 40, 45)
        self.selected_color = (70, 100, 140)
        self.hover_color = (55, 55, 65)
        self.text_color = (220, 220, 220)
        self.dim_text = (140, 140, 140)
        self.char_color = (100, 180, 255)
        self.anim_color = (180, 220, 180)
        
        # Grid settings
        self.show_grid = True
        self.grid_size = 32
        self.grid_color = (80, 80, 85)
        self.axis_x_color = (255, 100, 100)
        self.axis_y_color = (100, 255, 100)
        
        # Scaling mode: 'sharp' (nearest neighbor) or 'smooth' (bilinear)
        self.scale_mode = 'sharp'  # Sharp by default - preserves upscaled detail
        
        # Camera
        self.camera = Camera()
        self.is_panning = False
        self.pan_start = None
        self.auto_fit = True  # Auto-fit sprite to screen on load
        self.fit_margin = 0.8  # 80% of available space
        
        # State
        self.directory = Path(directory)
        self.tree = scan_directory(directory)
        self.expanded = set()
        self.selected_file = None
        self.hovered_item = None
        self.all_files = []  # Flat list of all files for navigation
        
        # Animation state
        self.frames = []           # Original PIL frames
        self.surfaces = []         # Original pygame surfaces
        self.upscaled_frames = []  # Upscaled PIL frames (cached)
        self.upscaled_surfaces = []  # Upscaled pygame surfaces (cached)
        self.frame_idx = 0
        self.info = None
        self.fps = 30
        self.paused = False
        self.frame_timer = 0
        
        # Upscaling - 8x for high zoom quality with sharpening
        # Use AI upscaler if available, CUDA upscaler as fallback, then PIL
        if UPSCALE_AVAILABLE and UPSCALE_METHOD == "RealESRGAN":
            self.upscaler = AIUpscaler(scale=4)  # Real-ESRGAN max is 4x
            print("Using Real-ESRGAN AI upscaling (4x + sharpen)")
            self.model_load_thread = threading.Thread(target=self.upscaler.load_model, daemon=True)
            self.model_load_thread.start()
        elif UPSCALE_AVAILABLE and UPSCALE_METHOD == "CUDA_LANCZOS":
            self.upscaler = CUDAUpscaler(scale=8)  # 8x for CUDA bicubic
            print("Using CUDA GPU upscaling (8x + sharpen)")
            self.model_load_thread = threading.Thread(target=self.upscaler.load_model, daemon=True)
            self.model_load_thread.start()
        else:
            self.upscaler = PILUpscaler(scale=8)  # 8x for PIL
            print("Using PIL CPU upscaling (8x + sharpen)")
        
        self.use_upscaled = True  # ON by default
        self.upscale_progress = 0
        self.upscale_total = 0
        self.upscaling_active = False
        self.upscale_queue = queue.Queue()
        self.upscale_thread = None
        self.auto_upscale = True  # Automatically upscale when loading
        
        # Build flat list for rendering
        self.rebuild_items()
        
        # Stats
        total_chars = len(self.tree)
        total_anims = sum(len(anims) for anims in self.tree.values())
        total_files = sum(len(files) for anims in self.tree.values() for files in anims.values())
        print(f"Loaded: {total_chars} characters, {total_anims} animations, {total_files} sprite sheets")
        
        # Preselect file if specified
        if preselect_file:
            self.load_sprite(Path(preselect_file))
            if self.info:
                self.expanded.add(self.info['char'])
                self.rebuild_items()
    
    def rebuild_items(self):
        """Build flat list of tree items - simplified structure"""
        self.items = []
        self.all_files = []  # Flat list of all files for prev/next navigation
        
        # Build flat file list for navigation (always includes all files)
        for char in sorted(self.tree.keys()):
            for anim in sorted(self.tree[char].keys()):
                for f in self.tree[char][anim]:
                    self.all_files.append(f)
        
        # Build visible tree items
        for char in sorted(self.tree.keys()):
            # Count total files for this character
            total_files = sum(len(files) for files in self.tree[char].values())
            self.items.append(('char', char, total_files))
            
            if char in self.expanded:
                # Flatten: show each animation directly (skip the animation grouping)
                for anim in sorted(self.tree[char].keys()):
                    for f in self.tree[char][anim]:
                        info = parse_sprite_sheet(f)
                        self.items.append(('file', f, info))
    
    def fit_to_screen(self):
        """Fit current sprite to screen with margin"""
        if not self.info:
            return
        
        preview_w = self.screen_w - self.tree_width - 40
        preview_h = self.screen_h - 200
        
        sprite_w = self.info['w']
        sprite_h = self.info['h']
        
        if sprite_w <= 0 or sprite_h <= 0:
            return
        
        # Calculate zoom to fit with margin
        zoom_w = (preview_w * self.fit_margin) / sprite_w
        zoom_h = (preview_h * self.fit_margin) / sprite_h
        
        self.camera.zoom = min(zoom_w, zoom_h, 8.0)  # Cap at 8x
        self.camera.zoom = max(self.camera.zoom, 0.25)  # Min 0.25x
        self.camera.offset_x = 0
        self.camera.offset_y = 0
    
    def goto_animation(self, delta):
        """Go to next (delta=1) or previous (delta=-1) animation"""
        if not self.all_files or not self.selected_file:
            return
        
        try:
            current_idx = self.all_files.index(self.selected_file)
        except ValueError:
            current_idx = 0
        
        new_idx = (current_idx + delta) % len(self.all_files)
        new_file = self.all_files[new_idx]
        
        # Expand the character containing this file
        info = parse_sprite_sheet(new_file)
        if info:
            self.expanded.add(info['char'])
            self.rebuild_items()
        
        self.load_sprite(new_file)
    
    def load_sprite(self, filepath):
        """Load a sprite sheet"""
        # Stop any current upscaling
        self.upscaling_active = False
        
        self.frames, self.info = extract_frames(filepath)
        self.surfaces = [pil_to_pygame(f) for f in self.frames] if self.frames else []
        self.upscaled_frames = []
        self.upscaled_surfaces = []
        self.frame_idx = 0
        self.selected_file = filepath
        self.upscale_progress = 0
        
        self.camera.reset()
        
        # Auto-fit to screen
        if self.auto_fit and self.info:
            self.fit_to_screen()
        
        if self.info:
            title = f"{self.info['char']} - {self.info['anim']} ({len(self.frames)} frames)"
            pygame.display.set_caption(title)
        
        # Auto-start upscaling if enabled and model ready
        if self.auto_upscale and self.upscaler.ready and self.frames:
            self.start_upscaling()
    
    def start_upscaling(self):
        """Start background upscaling of all frames"""
        if not self.upscaler.ready or not self.frames:
            return
        
        if self.upscaling_active:
            return
        
        # Clear previous upscaled data
        self.upscaled_frames = [None] * len(self.frames)
        self.upscaled_surfaces = [None] * len(self.frames)
        self.upscale_progress = 0
        self.upscale_total = len(self.frames)
        self.upscaling_active = True
        
        # Clear queue
        while not self.upscale_queue.empty():
            try:
                self.upscale_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start background thread
        self.upscale_thread = threading.Thread(target=self._upscale_worker, daemon=True)
        self.upscale_thread.start()
    
    def _upscale_worker(self):
        """Background worker for upscaling frames"""
        total = len(self.frames)
        print(f"Starting upscale of {total} frames using {type(self.upscaler).__name__}...")
        
        for i, frame in enumerate(self.frames):
            if not self.upscaling_active:
                print("Upscaling cancelled")
                break
            
            upscaled = self.upscaler.upscale_pil(frame)
            if upscaled:
                self.upscaled_frames[i] = upscaled
                self.upscale_queue.put((i, upscaled))
                # Only print every 10 frames or first/last
                if i == 0 or i == total - 1 or (i + 1) % 10 == 0:
                    print(f"Upscaled {i+1}/{total}: {frame.size} -> {upscaled.size}")
            else:
                print(f"Frame {i+1} FAILED")
            
            self.upscale_progress = i + 1
        
        self.upscaling_active = False
        success = sum(1 for s in self.upscaled_frames if s is not None)
        print(f"Upscaling complete! {success}/{total} frames processed")
    
    def process_upscale_queue(self):
        """Process completed upscales from background thread"""
        processed = 0
        while not self.upscale_queue.empty() and processed < 20:  # Process more per frame
            try:
                idx, pil_img = self.upscale_queue.get_nowait()
                self.upscaled_surfaces[idx] = pil_to_pygame(pil_img)
                processed += 1
            except queue.Empty:
                break
    
    def handle_tree_click(self, pos):
        if pos[0] > self.tree_width:
            return
        
        y = 10 - self.scroll_y
        for item in self.items:
            item_type = item[0]
            item_h = 24 if item_type == 'char' else 22
            
            if y <= pos[1] < y + item_h:
                if item_type == 'char':
                    char_name = item[1]
                    if char_name in self.expanded:
                        self.expanded.remove(char_name)
                    else:
                        self.expanded.add(char_name)
                    self.rebuild_items()
                elif item_type == 'file':
                    filepath = item[1]
                    self.load_sprite(filepath)
                return
            y += item_h
    
    def get_hovered_item(self, pos):
        if pos[0] > self.tree_width:
            return None
        
        y = 10 - self.scroll_y
        for item in self.items:
            item_type = item[0]
            item_h = 24 if item_type == 'char' else 22
            
            if y <= pos[1] < y + item_h:
                if item_type == 'char':
                    return ('char', item[1], item[2])
                else:
                    return ('file', item[1], item[2])
            y += item_h
        return None
    
    def draw_tree(self):
        pygame.draw.rect(self.screen, self.tree_bg, (0, 0, self.tree_width, self.screen_h))
        
        y = 10 - self.scroll_y
        max_y = 10
        
        for item in self.items:
            item_type = item[0]
            
            if item_type == 'char':
                char_name = item[1]
                file_count = item[2]
                
                rect = pygame.Rect(0, y, self.tree_width, 24)
                if self.hovered_item and self.hovered_item[0] == 'char' and self.hovered_item[1] == char_name:
                    pygame.draw.rect(self.screen, self.hover_color, rect)
                
                expanded = char_name in self.expanded
                prefix = "[-] " if expanded else "[+] "
                
                if 0 <= y < self.screen_h:
                    self.font.render_to(self.screen, (8, y + 4), prefix + char_name, self.char_color)
                    self.font_small.render_to(self.screen, (self.tree_width - 35, y + 6), f"({file_count})", self.dim_text)
                
                y += 24
                
            elif item_type == 'file':
                filepath = item[1]
                info = item[2]
                
                rect = pygame.Rect(20, y, self.tree_width - 20, 22)
                is_selected = filepath == self.selected_file
                is_hovered = self.hovered_item and self.hovered_item[0] == 'file' and self.hovered_item[1] == filepath
                
                if is_selected:
                    pygame.draw.rect(self.screen, self.selected_color, rect)
                elif is_hovered:
                    pygame.draw.rect(self.screen, self.hover_color, rect)
                
                if 0 <= y < self.screen_h:
                    if info:
                        # Show animation name and specs on same line
                        anim_name = info['anim']
                        specs = f"{info['w']}x{info['h']} g{info['gap']}"
                        self.font_small.render_to(self.screen, (28, y + 4), anim_name, self.anim_color)
                        self.font_small.render_to(self.screen, (self.tree_width - 85, y + 4), specs, self.dim_text)
                    else:
                        self.font_small.render_to(self.screen, (28, y + 4), filepath.stem[:25], self.text_color)
                
                y += 22
            
            max_y = y
        
        self.max_scroll = max(0, max_y - self.screen_h + 50)
        
        if self.max_scroll > 0:
            bar_h = max(30, (self.screen_h / (max_y + self.screen_h)) * self.screen_h)
            bar_y = (self.scroll_y / self.max_scroll) * (self.screen_h - bar_h) if self.max_scroll > 0 else 0
            pygame.draw.rect(self.screen, (60, 60, 70), (self.tree_width - 8, bar_y, 6, bar_h), border_radius=3)
        
        pygame.draw.line(self.screen, (60, 60, 70), (self.tree_width, 0), (self.tree_width, self.screen_h), 2)
    
    def draw_grid(self, preview_rect, center_x, center_y):
        if not self.show_grid:
            return
        
        left, top = self.camera.screen_to_world(preview_rect.left, preview_rect.top, center_x, center_y)
        right, bottom = self.camera.screen_to_world(preview_rect.right, preview_rect.bottom, center_x, center_y)
        
        grid_size = self.grid_size
        
        start_x = int(left // grid_size) * grid_size
        x = start_x
        while x <= right:
            sx, _ = self.camera.world_to_screen(x, 0, center_x, center_y)
            if preview_rect.left <= sx <= preview_rect.right:
                pygame.draw.line(self.screen, self.grid_color, 
                               (sx, preview_rect.top), (sx, preview_rect.bottom), 1)
            x += grid_size
        
        start_y = int(top // grid_size) * grid_size
        y = start_y
        while y <= bottom:
            _, sy = self.camera.world_to_screen(0, y, center_x, center_y)
            if preview_rect.top <= sy <= preview_rect.bottom:
                pygame.draw.line(self.screen, self.grid_color,
                               (preview_rect.left, sy), (preview_rect.right, sy), 1)
            y += grid_size
        
        _, axis_y = self.camera.world_to_screen(0, 0, center_x, center_y)
        if preview_rect.top <= axis_y <= preview_rect.bottom:
            pygame.draw.line(self.screen, self.axis_x_color,
                           (preview_rect.left, axis_y), (preview_rect.right, axis_y), 2)
        
        axis_x, _ = self.camera.world_to_screen(0, 0, center_x, center_y)
        if preview_rect.left <= axis_x <= preview_rect.right:
            pygame.draw.line(self.screen, self.axis_y_color,
                           (axis_x, preview_rect.top), (axis_x, preview_rect.bottom), 2)
    
    def scale_surface(self, surf, target_w, target_h, is_upscaled=False):
        """Scale surface using appropriate method"""
        if target_w <= 0 or target_h <= 0:
            return None
        
        src_w, src_h = surf.get_size()
        
        # If we're scaling DOWN an upscaled image, use smooth scaling
        # This preserves the anti-aliased quality of the upscale
        if is_upscaled and (target_w < src_w or target_h < src_h):
            return pygame.transform.smoothscale(surf, (target_w, target_h))
        
        # Otherwise use the selected mode
        if self.scale_mode == 'sharp':
            return pygame.transform.scale(surf, (target_w, target_h))
        else:
            return pygame.transform.smoothscale(surf, (target_w, target_h))
    
    def draw_preview(self):
        preview_x = self.tree_width + 10
        preview_w = self.screen_w - self.tree_width - 20
        preview_h = self.screen_h - 160
        preview_rect = pygame.Rect(preview_x, 60, preview_w, preview_h)
        
        pygame.draw.rect(self.screen, (35, 35, 40), preview_rect)
        
        center_x = preview_x + preview_w // 2
        center_y = 60 + preview_h // 2
        
        self.draw_grid(preview_rect, center_x, center_y)
        
        if not self.surfaces:
            self.font.render_to(self.screen, (preview_x + 20, 90), 
                "Select a sprite sheet from the tree", self.dim_text)
            self.font_small.render_to(self.screen, (preview_x + 20, 115),
                "Click ▶ to expand characters, then select an animation file", self.dim_text)
            self.draw_controls_help(preview_x)
            return
        
        # Header info
        if self.info:
            # Show current animation index
            try:
                anim_idx = self.all_files.index(self.selected_file) + 1 if self.selected_file and self.all_files else 0
                total_anims = len(self.all_files)
            except ValueError:
                anim_idx, total_anims = 0, 0
            
            header = f"{self.info['char']} - {self.info['anim']} [{anim_idx}/{total_anims}]"
            self.font.render_to(self.screen, (preview_x, 15), header, self.char_color)
            
            details = f"Size: {self.info['w']}x{self.info['h']} | Gap: {self.info['gap']} | Offset: {self.info['offset']}"
            self.font_small.render_to(self.screen, (preview_x, 38), details, self.dim_text)
        
        # Check if we have upscaled version for current frame
        have_upscaled = (self.use_upscaled and 
                        self.upscaled_surfaces and 
                        self.frame_idx < len(self.upscaled_surfaces) and 
                        self.upscaled_surfaces[self.frame_idx] is not None)
        
        # Original frame dimensions
        orig_w = self.info['w'] if self.info else self.surfaces[self.frame_idx].get_width()
        orig_h = self.info['h'] if self.info else self.surfaces[self.frame_idx].get_height()
        
        # Show source indicator
        if have_upscaled:
            up_surf = self.upscaled_surfaces[self.frame_idx]
            src_text = f"Source: UPSCALED {up_surf.get_width()}x{up_surf.get_height()}"
            src_color = (100, 255, 100)
        else:
            src_text = f"Source: Original {orig_w}x{orig_h}"
            src_color = (255, 200, 100)
        self.font_small.render_to(self.screen, (preview_x + preview_w - 250, 15), src_text, src_color)
        
        # Target display size based on zoom
        target_w = int(orig_w * self.camera.zoom)
        target_h = int(orig_h * self.camera.zoom)
        
        if target_w > 0 and target_h > 0:
            if have_upscaled:
                # Use upscaled surface (4x resolution)
                surf = self.upscaled_surfaces[self.frame_idx]
                scaled_surf = self.scale_surface(surf, target_w, target_h, is_upscaled=True)
            else:
                # Use original surface
                surf = self.surfaces[self.frame_idx]
                scaled_surf = self.scale_surface(surf, target_w, target_h, is_upscaled=False)
            
            if scaled_surf:
                # Position centered at origin
                world_x = -orig_w // 2
                world_y = -orig_h // 2
                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y, center_x, center_y)
                
                self.screen.set_clip(preview_rect)
                self.screen.blit(scaled_surf, (screen_x, screen_y))
                self.screen.set_clip(None)
        
        self.draw_frame_strip(preview_x, preview_w)
        self.draw_controls_help(preview_x)
    
    def draw_frame_strip(self, preview_x, preview_w):
        if not self.surfaces:
            return
        
        strip_y = self.screen_h - 130
        
        # Frame counter and status
        frame_text = f"Frame {self.frame_idx + 1}/{len(self.frames)}"
        self.font.render_to(self.screen, (preview_x, self.screen_h - 80), frame_text, self.text_color)
        
        status = "⏸ PAUSED" if self.paused else f"▶ {self.fps} FPS"
        self.font.render_to(self.screen, (preview_x + 150, self.screen_h - 80), status, 
            (255, 200, 100) if self.paused else (100, 255, 150))
        
        # Zoom level
        zoom_text = f"Zoom: {self.camera.zoom:.1f}x"
        self.font.render_to(self.screen, (preview_x + 280, self.screen_h - 80), zoom_text, (180, 180, 255))
        
        # Auto-fit indicator
        fit_text = "Fit:ON" if self.auto_fit else "Fit:OFF"
        fit_color = (100, 255, 100) if self.auto_fit else (150, 150, 150)
        self.font.render_to(self.screen, (preview_x + 380, self.screen_h - 80), fit_text, fit_color)
        
        # Scale mode
        mode_text = f"[{'Sharp' if self.scale_mode == 'sharp' else 'Smooth'}]"
        self.font.render_to(self.screen, (preview_x + 460, self.screen_h - 80), mode_text, (200, 200, 100))
        
        # Sharpen level
        sharpen = getattr(self.upscaler, 'sharpen_amount', 0)
        sharpen_text = f"Sharpen: {sharpen:.1f}"
        self.font.render_to(self.screen, (preview_x + 550, self.screen_h - 80), sharpen_text, (255, 180, 100))
        
        # Upscale status
        upscaler_type = type(self.upscaler).__name__
        scale = self.upscaler.scale if hasattr(self.upscaler, 'scale') else 4
        if hasattr(self.upscaler, 'loading') and self.upscaler.loading:
            status_text = self.upscaler.status_message
            status_color = (255, 255, 100)
        elif self.upscaling_active:
            pct = int(100 * self.upscale_progress / max(1, self.upscale_total))
            status_text = f"Upscaling: {pct}%"
            status_color = (100, 200, 255)
        elif self.use_upscaled and self.upscaled_surfaces and any(self.upscaled_surfaces):
            if upscaler_type == "AIUpscaler":
                status_text = f"AI {scale}x Enhanced"
                status_color = (100, 255, 100)
            elif upscaler_type == "CUDAUpscaler":
                status_text = f"CUDA {scale}x Sharp"
                status_color = (150, 255, 150)
            else:
                status_text = f"PIL {scale}x Sharp"
                status_color = (200, 200, 100)
        elif self.upscaler.ready:
            status_text = f"{upscaler_type[:4]} {scale}x Ready (U)"
            status_color = (150, 150, 150)
        else:
            status_text = self.upscaler.status_message if hasattr(self.upscaler, 'status_message') else "Loading..."
            status_color = (200, 200, 100)
        
        self.font.render_to(self.screen, (preview_x + 680, self.screen_h - 80), status_text, status_color)
        
        # Thumbnail strip
        thumb_size = 40
        max_thumbs = min(len(self.surfaces), (preview_w - 20) // (thumb_size + 4))
        
        start_idx = max(0, self.frame_idx - max_thumbs // 2)
        if start_idx + max_thumbs > len(self.surfaces):
            start_idx = max(0, len(self.surfaces) - max_thumbs)
        
        for i in range(min(max_thumbs, len(self.surfaces))):
            idx = start_idx + i
            if idx >= len(self.surfaces):
                break
            
            # Use upscaled if available
            if (self.use_upscaled and self.upscaled_surfaces and 
                idx < len(self.upscaled_surfaces) and self.upscaled_surfaces[idx]):
                thumb = pygame.transform.smoothscale(self.upscaled_surfaces[idx], (thumb_size, thumb_size))
            else:
                thumb = pygame.transform.smoothscale(self.surfaces[idx], (thumb_size, thumb_size))
            
            tx = preview_x + i * (thumb_size + 4)
            
            if idx == self.frame_idx:
                pygame.draw.rect(self.screen, (100, 180, 255), (tx - 2, strip_y - 2, thumb_size + 4, thumb_size + 4), 2)
            
            self.screen.blit(thumb, (tx, strip_y))
    
    def draw_controls_help(self, preview_x):
        controls = [
            "SPACE: pause | Left/Right: step | Up/Down: FPS | N/PgDn: next | P/PgUp: prev | F: fit",
            "G: grid | +/-: grid size | U: upscale | T: sharp/smooth | [ ]: sharpen | Q: reset | ESC: quit"
        ]
        for i, line in enumerate(controls):
            self.font_small.render_to(self.screen, (preview_x, self.screen_h - 50 + i * 18), line, self.dim_text)
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0
            
            # Process upscale queue
            self.process_upscale_queue()
            
            # Check if model just became ready and we have frames waiting
            if self.upscaler.ready and self.auto_upscale and self.frames and not self.upscaled_surfaces:
                self.start_upscaling()
            
            # Animation update
            if not self.paused and self.surfaces:
                self.frame_timer += dt
                if self.frame_timer >= 1.0 / self.fps:
                    self.frame_timer = 0
                    self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            
            mouse_pos = pygame.mouse.get_pos()
            self.hovered_item = self.get_hovered_item(mouse_pos)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_w, self.screen_h = event.w, event.h
                    self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_tree_click(event.pos)
                    elif event.button in (2, 3):
                        if event.pos[0] > self.tree_width:
                            self.is_panning = True
                            self.pan_start = event.pos
                    elif event.button == 4:
                        if event.pos[0] <= self.tree_width:
                            self.scroll_y = max(0, self.scroll_y - self.scroll_speed)
                        else:
                            preview_x = self.tree_width + 10
                            preview_w = self.screen_w - self.tree_width - 20
                            preview_h = self.screen_h - 160
                            center_x = preview_x + preview_w // 2
                            center_y = 60 + preview_h // 2
                            self.camera.zoom_at(1.25, event.pos[0], event.pos[1], center_x, center_y)
                    elif event.button == 5:
                        if event.pos[0] <= self.tree_width:
                            self.scroll_y = min(self.max_scroll, self.scroll_y + self.scroll_speed)
                        else:
                            preview_x = self.tree_width + 10
                            preview_w = self.screen_w - self.tree_width - 20
                            preview_h = self.screen_h - 160
                            center_x = preview_x + preview_w // 2
                            center_y = 60 + preview_h // 2
                            self.camera.zoom_at(1/1.25, event.pos[0], event.pos[1], center_x, center_y)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (2, 3):
                        self.is_panning = False
                        self.pan_start = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.is_panning and self.pan_start:
                        dx = event.pos[0] - self.pan_start[0]
                        dy = event.pos[1] - self.pan_start[1]
                        self.camera.pan(dx, dy)
                        self.pan_start = event.pos
                        
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.upscaling_active = False
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_LEFT:
                        if self.surfaces:
                            self.frame_idx = (self.frame_idx - 1) % len(self.frames)
                    elif event.key == pygame.K_RIGHT:
                        if self.surfaces:
                            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
                    elif event.key == pygame.K_UP:
                        self.fps = min(60, self.fps + 2)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(2, self.fps - 2)
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_q:
                        self.camera.reset()
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.grid_size = min(256, self.grid_size * 2)
                    elif event.key == pygame.K_MINUS:
                        self.grid_size = max(8, self.grid_size // 2)
                    elif event.key == pygame.K_1:
                        self.camera.zoom = 1.0
                    elif event.key == pygame.K_2:
                        self.camera.zoom = 2.0
                    elif event.key == pygame.K_3:
                        self.camera.zoom = 3.0
                    elif event.key == pygame.K_4:
                        self.camera.zoom = 4.0
                    elif event.key == pygame.K_t:
                        # Toggle scale mode
                        self.scale_mode = 'smooth' if self.scale_mode == 'sharp' else 'sharp'
                    elif event.key == pygame.K_f:
                        # Toggle auto-fit / fit now
                        self.auto_fit = not self.auto_fit
                        if self.auto_fit:
                            self.fit_to_screen()
                    elif event.key == pygame.K_PAGEDOWN or event.key == pygame.K_n:
                        # Next animation
                        self.goto_animation(1)
                    elif event.key == pygame.K_PAGEUP or event.key == pygame.K_p:
                        # Previous animation
                        self.goto_animation(-1)
                    elif event.key == pygame.K_u:
                        # Toggle/start AI upscaling
                        if self.upscaler.ready and self.frames:
                            if not self.upscaled_surfaces or not any(self.upscaled_surfaces):
                                self.start_upscaling()
                                self.use_upscaled = True
                            else:
                                self.use_upscaled = not self.use_upscaled
                    elif event.key == pygame.K_LEFTBRACKET:
                        # Decrease sharpening
                        if hasattr(self.upscaler, 'sharpen_amount'):
                            self.upscaler.sharpen_amount = max(0, self.upscaler.sharpen_amount - 0.25)
                            print(f"Sharpen: {self.upscaler.sharpen_amount:.2f}")
                            # Re-upscale with new setting
                            if self.upscaled_surfaces and any(self.upscaled_surfaces):
                                self.start_upscaling()
                    elif event.key == pygame.K_RIGHTBRACKET:
                        # Increase sharpening
                        if hasattr(self.upscaler, 'sharpen_amount'):
                            self.upscaler.sharpen_amount = min(3.0, self.upscaler.sharpen_amount + 0.25)
                            print(f"Sharpen: {self.upscaler.sharpen_amount:.2f}")
                            # Re-upscale with new setting
                            if self.upscaled_surfaces and any(self.upscaled_surfaces):
                                self.start_upscaling()
            
            # Draw
            self.screen.fill(self.bg_color)
            self.draw_tree()
            self.draw_preview()
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r'G:\SteamLibrary\steamapps\common\Guilty Hell\exported\Texture2D'
    path = Path(path)
    
    if path.is_file():
        directory = path.parent
        preselect = path
    else:
        directory = path
        preselect = None
    
    viewer = SpriteViewer(directory, preselect)
    viewer.run()