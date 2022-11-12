import taichi as ti
from model.PBF import *
import os, shutil
import cv2
import time

ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)

def generate_video(img_dir, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("fluid_simulation_pbf.mp4", fourcc, fps, GUI_Resolution)
    image_file_list = sorted(os.listdir(img_dir))
    for path in image_file_list:
        img = cv2.imread(os.path.join(img_dir, path))
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":
    
    sph_solver = PBF()
    gui = ti.GUI(background_color=Background_Color, show_gui=False, res=GUI_Resolution)
    cnt = 0
    start_time = time.time()
    while gui.running:
        for i in range(1):
            sph_solver.solve()
        fluid_pos, boundary_pos = sph_solver.scene.get_particle_pos()
        gui.circles(pos=fluid_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Particle_Color)
        gui.circles(pos=boundary_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Boundary_Color)
        cnt += 1
        if cnt > 1000:
            break
        if not os.path.exists('log'):
            os.mkdir('log')
        filename = f'log/frame_{cnt:05d}.png'
        gui.show(filename)
    end_time = time.time()
    print(f'program runtime: {end_time-start_time}s')

    # generate video
    generate_video('log', 200)
    shutil.rmtree('log')
