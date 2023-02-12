import pyvista as pv

mesh = pv.read('../../data/cases/dataset1/Design_1/Mapped_Blade_Surface.vtp')
print(mesh.points.shape)

#plot using pyvista
pl = pv.Plotter(window_size = [400,400])
pl.background_color = 'w'
pl.add_points(mesh.points[(896, 954),:], color = 'red',point_size = 5)
pl.add_mesh(mesh,color = 'gray',show_edges = True, opacity=1)# scalars = ?
#glyphs = mesh.glyph(orient='vectors', scale='scalars', factor=0.003, geom=geom)
#pl.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm')
pl.enable_anti_aliasing()
pl.show(jupyter_backend='panel')


# run it using geoheaven