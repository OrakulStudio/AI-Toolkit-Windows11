import os
import re
import matplotlib
matplotlib.use('TkAgg')  # GUI бэкенд Tkinter

# Отключаем нижнюю тулбар-панель кнопок matplotlib
matplotlib.rcParams['toolbar'] = 'None'

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- НАСТРОЙКИ ---
LOG_FILE = "Log_r128f3kleintest.txt"
UPDATE_MS = 3000  # Обновление каждые 3 секунды

# --- ИНИЦИАЛИЗА ОКНА ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 4.5))

# Стандартное имя окна Windows
try:
    fig.canvas.manager.window.wm_title('Orakul Viking Monitor')
except Exception:
    pass

fig.patch.set_facecolor('#0B0F14')
ax.set_facecolor('#0B0F14')

# Компактные отступы, чтобы график занимал всё полезное место
plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.1)

def animate(i):
    if not os.path.exists(LOG_FILE):
        return

    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Парсим значения loss из лога tqdm
        found = re.findall(r'loss:\s*([0-9\.]+(?:e[-+]?\d+)?)', content, re.IGNORECASE)
        
        valid_losses = []
        for x in found:
            try:
                v = float(x)
                if 0.0 < v < 4.0:
                    valid_losses.append(v)
            except ValueError:
                pass
                
        losses = valid_losses
    except Exception as e:
        print(f"Ошибка чтения: {e}")
        return

    if losses:
        ax.clear()
        ax.set_facecolor('#0B0F14')
        
        steps = range(len(losses))
        
        # Экспоненциальное сглаживание (EMA)
        alpha = 0.04
        smoothed = []
        curr_ema = losses[0]
        for val in losses:
            curr_ema = alpha * val + (1 - alpha) * curr_ema
            smoothed.append(curr_ema)

        # Автомасштаб Y по рабочим значениям
        y_min = max(0, min(smoothed) * 0.85)
        y_max = max(smoothed) * 1.15
        ax.set_ylim(y_min, y_max)

        # 1. Тончайший фоновый шум сырых лоссов
        ax.plot(steps, losses, color='#e48a0c', alpha=0.12, linewidth=0.5)
        
        # 2. Основная яркая кривая тренда
        ax.plot(steps, smoothed, color='#a1a1a1', linewidth=1.0)
        
        # 3. Легкая неоновая заливка под графиком
        ax.fill_between(steps, smoothed, y_min, color='#00FFFF', alpha=0.06)

        # Настройка сетки
        ax.grid(True, alpha=0.08, color='#e4bb62', linestyle='--')
        
        current_loss = losses[-1]
        avg_loss = smoothed[-1]
        title_text = f" Orakul Viking Monitor  |  Step: {len(losses)}  |  Loss: {current_loss:.4f}  |  EMA: {avg_loss:.4f}"
        
        ax.set_title(title_text, color='#e4bb62', fontsize=10, loc='left', fontfamily='monospace', fontweight='bold')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E293B')
            spine.set_linewidth(0.8)
            
        ax.tick_params(colors='#64748B', labelsize=8)

# --- ЗАПУСК ---
ani = animation.FuncAnimation(fig, animate, interval=UPDATE_MS)
plt.show()