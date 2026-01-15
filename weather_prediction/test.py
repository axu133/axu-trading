import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

year = 2025
month = 1
file_path = os.path.join("data", "era5_raw", str(year), f"era5_{year}_{month:02d}.nc")

ds = xr.open_dataset(file_path)

subset = ds.isel(valid_time=slice(0, 240))

temp_f = (subset['t2m'] - 273.15) * 9/5 + 32

u = subset['u10']
v = subset['v10']

skip = 5
sl_x = slice(None, None, skip)
sl_y = slice(None, None, skip)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linestyle=':')

ax.set_extent([-88, -72, 33, 49], crs=ccrs.PlateCarree())

mesh = temp_f.isel(valid_time=0).plot.pcolormesh(
    ax=ax, 
    transform=ccrs.PlateCarree(),
    cmap='coolwarm',
    add_colorbar=True,
    cbar_kwargs={'label': 'Temperature (°F)', 'orientation': 'horizontal', 'pad': 0.05},
    vmin=temp_f.min(), 
    vmax=temp_f.max(),
    zorder=0
)

X = subset.longitude.values[sl_x]
Y = subset.latitude.values[sl_y]
U_init = u.isel(valid_time=0)[sl_y, sl_x].values
V_init = v.isel(valid_time=0)[sl_y, sl_x].values

quiver = ax.quiver(
    X, Y, U_init, V_init,
    transform=ccrs.PlateCarree(),
    color='black',
    scale=400,
    width=0.003,
    zorder=1
)

title = ax.set_title("Initializing")

def update(frame):
    t_data = temp_f.isel(valid_time=frame).values.ravel()
    mesh.set_array(t_data)
    
    u_data = u.isel(valid_time=frame)[sl_y, sl_x].values
    v_data = v.isel(valid_time=frame)[sl_y, sl_x].values
    quiver.set_UVC(u_data, v_data)
    
    time_str = str(subset.valid_time.values[frame])[:16].replace('T', ' ')
    title.set_text(f"Forecast: {time_str}")
    
    return mesh, quiver, title

anim = FuncAnimation(fig, update, frames=len(subset.valid_time), interval=100, blit=False)

anim.save('weather_map.gif', writer='pillow', fps=8)