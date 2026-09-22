import time
import subprocess
from tqdm import tqdm
from . import orakul_studio

# --- 4 СООБЩЕНИЯ НА ВЕСЬ ПРОГОН (~2.5 ЧАСА) ---
PROMO_MESSAGES = [
    "[!] SUPER Rank 2048 at 168s/it Set offload to 0.91 and fly! — Thanks for using my code <°_°> 22.09.26.",
    "[!] Welcome to my Civitai:  https://civitai.com/user/ORAKUL_STUDIO.",
    "[!] Star the repo on GitHub / Hugging Face if this helped!",    
    "[!] Pure 8-bit pipeline — No OOM, No Compromise."
]

_last_gpu_check = 0
_gpu_str = ""
_promo_index = 0
_pending_promo_msg = None
_start_time = None

# Интервал 1800 секунд 30 Минут между сообщениями
PROMO_INTERVAL = 1800


def get_gpu_stats():
    global _last_gpu_check, _gpu_str
    cur_t = time.time()
    if cur_t - _last_gpu_check > 5:
        try:
            res = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw,clocks.gr', '--format=csv,noheader,nounits'],
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            ).strip().split(', ')
            if len(res) >= 2:
                _gpu_str = f"[{res[0]}W {res[1]}MHz] "
            else:
                _gpu_str = ""
        except Exception:
            _gpu_str = ""
        _last_gpu_check = cur_t
    return _gpu_str


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        orakul_studio.print_banner()
        super().__init__(*args, **kwargs)
        self.paused = False
        self.last_time = self._time()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.last_time = self._time()

    def unpause(self):
        if self.paused:
            self.paused = False
            cur_t = self._time()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, n=1):
        if not self.paused:
            global _pending_promo_msg
            # Безопасно печатаем отложенное сообщение прямо перед обновлением шагов
            if _pending_promo_msg:
                tqdm.write(_pending_promo_msg, file=self.fp)
                _pending_promo_msg = None
                
            super().update(n)

    @property
    def format_dict(self):
        global _promo_index, _pending_promo_msg, _start_time
        d = super().format_dict
        
        cur_t = time.time()
        if _start_time is None:
            _start_time = cur_t

        # Если еще остались сообщения и прошло нужное время с момента старта
        if _promo_index < len(PROMO_MESSAGES):
            elapsed_since_start = cur_t - _start_time
            target_time = _promo_index * PROMO_INTERVAL
            if elapsed_since_start >= target_time:
                _pending_promo_msg = PROMO_MESSAGES[_promo_index]
                _promo_index += 1

        stats = get_gpu_stats()
        if stats:
            d['prefix'] = f"{stats}{d.get('prefix', '')}"
        return d