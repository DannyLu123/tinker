import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick detected")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick name: {joystick.get_name()}")

DEAD_ZONE = 0.8  # 设置死区阈值

try:
    while True:
        pygame.event.pump()
        # 读取轴值
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        # 应用死区
        axes = [0.0 if abs(axis) < DEAD_ZONE else axis for axis in axes]
        print("Axes:", [f"{axis:.4f}" for axis in axes])
        
        # 读取按键
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        print("Buttons:", buttons)
        
        if buttons[0]:  # 按 A 键退出（Xbox 手柄的 A 键通常为 button 0）
            break
        
        time.sleep(0.1)  # 控制刷新频率，避免输出过快
except KeyboardInterrupt:
    print("Exiting...")
finally:
    pygame.quit()