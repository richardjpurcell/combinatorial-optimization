import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button



def main():
    # Load state data
    with open("states.json", "r") as f:
        all_states = json.load(f)

    # Dynamically calculate the grid size
    uncertainty_length = len(all_states[0]["uncertainty"])
    GRID_SIZE = int(np.sqrt(uncertainty_length))

    NUM_DRONES = len(all_states[0]["drones"])
    FADE_TIME = 5  # Number of time slots before cell turns black

    # Track fade levels for each cell
    fade_tracker = np.zeros((GRID_SIZE, GRID_SIZE))

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.2)  # Leave space for buttons
    unc_img = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap="gray", vmin=0, vmax=1)
    drone_lines = []
    drone_markers = []
    drone_paths = [[] for _ in range(NUM_DRONES)]
    colors = ["blue", "green", "magenta", "cyan"]

    for d in range(NUM_DRONES):
        line, = ax.plot([], [], color=colors[d % len(colors)], linewidth=2)
        marker, = ax.plot([], [], marker="o", color=colors[d % len(colors)], markersize=8)
        drone_lines.append(line)
        drone_markers.append(marker)

    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.grid(False)     # Disable the grid lines
    ax.invert_yaxis()

    # Animation state
    anim_running = True

    def toggle_animation(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
            play_button.label.set_text("Play")
        else:
            anim.event_source.start()
            anim_running = True
            play_button.label.set_text("Pause")

    def init_func():
        # Reset the heatmap, drone paths, and decorations
        unc_img.set_data(np.zeros((GRID_SIZE, GRID_SIZE)))
        for line in drone_lines:
            line.set_data([], [])
        for marker in drone_markers:
            marker.set_data([], [])
        return drone_lines + drone_markers + [unc_img]

    def update(frame_idx):
        nonlocal fade_tracker

        if frame_idx >= len(all_states):
            # Reset for next loop
            fade_tracker = np.zeros((GRID_SIZE, GRID_SIZE))
            for d in range(NUM_DRONES):
                drone_paths[d] = []
            unc_img.set_data(np.zeros((GRID_SIZE, GRID_SIZE)))
            ax.set_title("Restarting animation...")
            return drone_lines + drone_markers + [unc_img]

        # Load the current state
        state = all_states[frame_idx]
        visited_cells = [d["pos"] for d in state["drones"]]

        # Update fade tracker
        fade_tracker = np.maximum(fade_tracker - 1, 0)  # Decrement all cells
        for cell in visited_cells:
            row = cell // GRID_SIZE
            col = cell % GRID_SIZE
            fade_tracker[row, col] = FADE_TIME  # Reset luminance for visited cells

        # Generate grayscale values
        luminance = fade_tracker / FADE_TIME
        for cell in visited_cells:
            row = cell // GRID_SIZE
            col = cell % GRID_SIZE
            luminance[row, col] = 1.0  # Override visited cells to white

        unc_img.set_data(luminance)

        # Update drone paths and markers
        for d, drone in enumerate(state["drones"]):
            dpos = drone["pos"]
            row = dpos // GRID_SIZE
            col = dpos % GRID_SIZE

            # Extend path
            drone_paths[d].append((row, col))

            # Update line (path)
            xs = [c[1] for c in drone_paths[d]]  # col
            ys = [c[0] for c in drone_paths[d]]  # row
            drone_lines[d].set_data(xs, ys)

            # Update marker (current position)
            drone_markers[d].set_data([col], [row])

        ax.set_title(f"Time={state['time']}")
        return drone_lines + drone_markers + [unc_img]

    # Create the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(all_states) + 1,  # Extra frame for resetting
        init_func=init_func,
        blit=False,
        repeat=True,
        interval=500,  # 500 ms per frame
    )

    # Add Pause/Play button
    play_ax = plt.axes([0.4, 0.05, 0.2, 0.075])  # x, y, width, height
    play_button = Button(play_ax, "Pause", hovercolor="0.8")
    play_button.on_clicked(toggle_animation)

    plt.show()

if __name__ == "__main__":
    main()
