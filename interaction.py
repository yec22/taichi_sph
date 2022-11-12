import taichi as ti
import numpy as np
from model.WCSPH import *
import os

ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)


if __name__ == "__main__":

    if not os.path.exists('log'):
        os.mkdir('log')

    sph_solver = WCSPH()

    gui = ti.GUI(background_color=Background_Color, show_gui=True, res=GUI_Resolution)
    cnt = 0

    while gui.running:
        for i in range(100):
            sph_solver.solve()
        fluid_pos, boundary_pos = sph_solver.scene.get_particle_pos()
        gui.circles(pos=fluid_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Particle_Color)
        gui.circles(pos=boundary_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Boundary_Color)
        
        if gui.get_event(ti.GUI.SPACE):
            print("save results ...")
            writer = ti.tools.PLYWriter(num_vertices=sph_solver.scene.N[None])
            writer.add_vertex_pos(fluid_pos[:, 0], fluid_pos[:, 1], np.zeros_like(fluid_pos[:, 0]))
            writer.export_frame_ascii(cnt, f'log/frame_{cnt:06d}')

            filename = f'log/frame_{cnt:05d}.png'
            gui.show(filename)
        
        if gui.get_event(ti.GUI.ESCAPE):
            print("SPH simulation end ...")
            break

        cnt += 1
        gui.show()
