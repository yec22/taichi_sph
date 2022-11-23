import taichi as ti
import numpy as np
from model.WCSPH import *
import os

ti.init(arch=ti.cpu)

if __name__ == "__main__":
    # initialization
    if not os.path.exists('log'):
        os.mkdir('log')

    sph_solver = WCSPH()

    gui = ti.GUI(background_color=Background_Color, show_gui=True, res=GUI_Resolution)
    cnt = 0

    # gui loop
    while gui.running:
        mouse = gui.get_cursor_pos()
        gui.circle((mouse[0], mouse[1]), color=0xff0000, radius=7)

        # handle events
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'q':
                print("SPH simulation end ...")
                break
            elif gui.event.key in [ti.GUI.SPACE]:
                print("save results ...")
                writer = ti.tools.PLYWriter(num_vertices=sph_solver.scene.N[None])
                writer.add_vertex_pos(fluid_pos[:, 0], fluid_pos[:, 1], np.zeros_like(fluid_pos[:, 0]))
                writer.export_frame_ascii(cnt, f'log/frame_{cnt:08d}')
            elif gui.event.key in [ti.GUI.LMB]:
                mouse = gui.get_cursor_pos()
                particle_x = mouse[0] * GUI_Resolution[0] / Scale_Ratio
                particle_y = mouse[1] * GUI_Resolution[1] / Scale_Ratio
                if (particle_x - 0.5 > Support_Radius) and (particle_x + 0.5 < GUI_Resolution[0] / Scale_Ratio - Support_Radius) and (particle_y - 0.5 > Support_Radius) and (particle_y + 0.5 < GUI_Resolution[1] / Scale_Ratio - Support_Radius):
                    print("add particles ...", mouse[0], mouse[1])
                    sph_solver.scene.add_stuff(FLUID, [[particle_x-0.5, particle_x+0.5], [particle_y-0.5, particle_y+0.5]], [0.0, -10.0])

        # solve sph
        sph_solver.solve(cnt)
        fluid_pos, boundary_pos = sph_solver.scene.get_particle_pos()

        # visualization
        gui.circles(pos=fluid_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Particle_Color)
        gui.circles(pos=boundary_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Boundary_Color)

        cnt += 1
        gui.show()
