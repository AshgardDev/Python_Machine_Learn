# 2d动画
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # 设置画布
# fig, ax = plt.subplots()
# ### 这是第一桢的xy坐标
# x = np.linspace(0, 2*np.pi, 100)
# line, = ax.plot(x, np.sin(x))  # 初始折线图
#
# # 更新函数
# def update(frame):
#     line.set_ydata(np.sin(x + frame * 0.1))  # 更新 y 数据
#     return line,
#
# # 创建动画
# 重要参数（FuncAnimation）
# fig：要动画的 Figure 对象。
# func：更新函数（如 update），接收 frames 参数。
# frames：帧数，可以是整数（表示帧数）或可迭代对象（每一帧的值）。
# interval：帧之间的延迟（毫秒）。
# blit：是否只重绘变化部分（2D 动画推荐开启，3D 通常关闭）。
# repeat：动画是否循环（默认 True）。
## blit=True：优化性能，只重绘变化的部分。--2d适用
# ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
#
# plt.show()


## 3d动画 必须要先清除缓存
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # 设置 3D 画布
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 初始数据
# x = np.linspace(-2, 2, 20)
# y = np.linspace(-2, 2, 20)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(np.sqrt(X**2 + Y**2))
#
# # 初始曲面
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')
#
# # 更新函数
# def update(frame):
#     ##ax.clear()：每次更新时清除之前的曲面，因为 3D 图不像 2D 线条可以直接修改数据
#     ax.clear()  # 清除上一帧
#     Z = np.sin(np.sqrt(X**2 + Y**2) + frame * 0.1)  # 更新 Z 数据
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis')  # 重绘曲面
#     ax.set_zlim(-1.5, 1.5)  # 固定 z 轴范围
#     return surf,
#
# # def update(frame):
# #     ax.clear()
# #     X_new = X + 0.1 * np.sin(Y + frame * 0.1)  # 动态扭曲 X
# #     Z = np.sin(np.sqrt(X_new**2 + Y**2))
# #     surf = ax.plot_surface(X_new, Y, Z, cmap='viridis')
# #     ax.set_zlim(-1.5, 1.5)
# #     return surf,
#
# # 创建动画
# # blit=False：3D 动画通常不支持 blit，因为整个曲面需要重绘
# ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
#
# plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# x = np.random.randint(1, 100, size=100)
# y = np.random.randint(1, 100, size=100)
# c = np.random.randint(1, 100, size=100)
# s = np.random.randint(1, 100, size=100)
# s[s <0 ] = 0
#
#
# fig = plt.figure()
# im = plt.scatter(
#     x,
#     y,
#     s=s,
#     c = c,
#     cmap = "jet",
# )
#
# def update(frame):
#     im.set_sizes(s + frame*1)
#     return im,
#
# ani = FuncAnimation(fig, update, frames=100, interval=10, blit=False)
#
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as amin
# from matplotlib.animation import FuncAnimation
#
# n = 100
# balls = np.zeros(n, dtype=[
#     ('position', np.float32, 2),  ## 起始坐标
#     ('size', np.float32, 1),      ## 初始大小
#     ('growth', np.float32, 1),    ## 生长速度
#     ('color', np.float32, 4),     ## 颜色(rgba)
# ])
# balls['position'] = np.random.uniform(0, 1, (n, 2))
# balls['size'] = np.random.uniform(20, 100, (n, 1))
# balls['growth'] = np.random.uniform(5, 10, (n, 1))
# balls['color'] = np.random.uniform(0, 1, (n, 4))
#
# plt.figure('Animation')
# plt.title('Animation', fontsize=14)
#
# im = plt.scatter(
#     balls['position'][:, 0],
#     balls['position'][:, 1],
#     s=balls['size'][:, 0],
#     c=balls['color'],
# )
#
# def update(frame):
#     tmp = balls['size'][:,0]
#     balls['size'][:,0] = balls['size'][:,0] + balls['growth'][:,0]
#     tmp[tmp > 500] = 0
#     im.set_sizes(tmp)
#     # ## 这里必须返回元组对象!!!
#     return im,
#
# ## 这里一定要有个变量接收,否则动画不生效!!不会调用update函数
# anim = FuncAnimation(plt.gcf(), update, frames=100, interval=100, blit=True)
#
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as amin
from matplotlib.animation import FuncAnimation
from pyparsing import alphas

plt.figure('Signal')
plt.title('Signal', fontweight='bold', fontsize=14)
plt.xlim(0, 10)
plt.ylim(-3, 3)
plt.grid(linestyle=':', color='k', alpha=0.5)
## 这里返回的是个数组,一定要添加逗号,才能获取到绘图的对象!!
pl, = plt.plot([], [], color='red', label='Signal')
new_x = 0
def update(data):
    new_x, new_y = data
    x, y = pl.get_data()
    x = np.append(x, new_x)
    y = np.append(y, new_y)
    ## 重新设置数据源
    pl.set_data(x, y)
    if x[-1] > 10:
        plt.xlim(x[-1]-10, x[-1])
    return pl,

def y_generator():
    global new_x
    new_y = np.sin(2*np.pi*new_x) * np.exp(np.sin(0.2*np.pi*new_x))
    yield (new_x, new_y)
    new_x += 0.05

## 这里采用生成器,达到数据和绘图分割的目的
anim = FuncAnimation(plt.gcf(), update, y_generator, interval=20, cache_frame_data=False)

plt.tight_layout()
plt.show()