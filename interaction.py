import taichi as ti
import numpy as np
from model.WCSPH import *
import os

ti.init(arch=ti.cpu)


if __name__ == "__main__":

    if not os.path.exists('log'):
        os.mkdir('log')

    sph_solver = WCSPH()

    gui = ti.GUI(background_color=Background_Color, show_gui=True, res=GUI_Resolution)
    cnt = 0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'q':
                print("SPH simulation end ...")
                break
            elif gui.event.key in [ti.GUI.SPACE]:
                print("save results ...")
                writer = ti.tools.PLYWriter(num_vertices=sph_solver.scene.N[None])
                writer.add_vertex_pos(fluid_pos[:, 0], fluid_pos[:, 1], np.zeros_like(fluid_pos[:, 0]))
                writer.export_frame_ascii(cnt, f'log/frame_{cnt:08d}')

                filename = f'log/frame_{cnt:08d}.png'
                gui.show(filename)

        sph_solver.solve()
        fluid_pos, boundary_pos = sph_solver.scene.get_particle_pos()

        gui.circles(pos=fluid_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Particle_Color)
        gui.circles(pos=boundary_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Boundary_Color)
        
        mouse = gui.get_cursor_pos()
        gui.circle((mouse[0], mouse[1]), color=0xff0000, radius=7)
        if gui.is_pressed(ti.GUI.LMB):
            print("add particles ...")
            sph_solver.scene.add_stuff(FLUID, [[mouse[0], mouse[0]+1.0], [mouse[1], mouse[1]+1.0]], [0.0, -10.0])

        cnt += 1
        gui.show()
