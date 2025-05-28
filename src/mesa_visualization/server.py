import sys
import os
import numpy as np

# Prepend the 'src' directory to sys.path to ensure modules like room_types are found
# and to help AudioEnv resolve its internal imports if it expects 'src' to be a root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_SRC_DIR not in sys.path:
    sys.path.insert(0, PARENT_SRC_DIR)

# Also add the project root (grandparent of server.py, which is parent of src)
# This is often needed if AudioEnv or its dependencies (like constants.py)
# use paths relative to the project root, or if AudioEnv does something like sys.path.append("../../")
# from its own location (src/audio_room/envs/) which would point to the project root.
GRANDPARENT_PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if GRANDPARENT_PROJECT_DIR not in sys.path:
    sys.path.insert(0, GRANDPARENT_PROJECT_DIR)


import mesa
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ContinuousCanvas

from .agents import NavigatingAgent, SourceAgent # Relative import for agents in the same package
from .model import AudioMesaModel # Relative import for model in the same package

# This should now work due to sys.path modification above
try:
    import room_types
except ImportError:
    print("Failed to import room_types. Check sys.path and file locations.", file=sys.stderr)
    # As a fallback, if room_types is in src/ let's try one more specific path add for it.
    # This indicates a potential deeper issue with how modules are structured or expected.
    if os.path.join(PARENT_SRC_DIR, 'room_types.py') not in sys.path and \
       os.path.exists(os.path.join(PARENT_SRC_DIR, 'room_types.py')):
        print(f"Attempting to add {PARENT_SRC_DIR} for room_types.py specifically", file=sys.stderr)
        # This was already done with PARENT_SRC_DIR, so this is redundant but for clarity.
    raise # Re-raise the import error if it still fails

CANVAS_WIDTH_PX = 500
CANVAS_HEIGHT_PX = 500

def agent_portrayal(agent):
    if agent is None:
        return

    portrayal = {"Filled": "true", "Layer": 0}

    if isinstance(agent, NavigatingAgent):
        portrayal["Shape"] = "arrowHead"
        portrayal["Color"] = "blue"
        portrayal["scale"] = 0.8
        portrayal["heading_x"] = np.cos(agent.angle)
        portrayal["heading_y"] = np.sin(agent.angle)
        portrayal["r"] = 0.5 # Base size for arrowHead, or general radius
    elif isinstance(agent, SourceAgent):
        portrayal["Shape"] = "circle"
        # Example of dynamic color based on source_type:
        color_map = {"phone": "green", "siren": "yellow", "car_horn": "orange", "unknown": "red"}
        portrayal["Color"] = color_map.get(str(agent.source_type).lower(), "red") # Make it case-insensitive
        portrayal["r"] = 0.3
    else:
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "grey"
        portrayal["w"] = 0.2
        portrayal["h"] = 0.2
        
    return portrayal

# Define a default room and sources for the visualization server
# Paths for source_folders_dict should be relative to the project root,
# as AudioEnv or AudioSignal likely resolves them from there.
# If server.py is in src/mesa_visualization/, then GRANDPARENT_PROJECT_DIR is the project root.
# So, 'sounds/phone/' would correctly point to 'PROJECT_ROOT/sounds/phone/'.
default_room = room_types.ShoeBox(x_length=10, y_length=10)
default_source_folders = {
    'sounds/phone/': 1,
    'sounds/siren/': 1
    # Add more source types and their paths as needed
}

# Temporary model instance to get canvas dimensions correctly from the model's space
# This ensures the canvas matches the actual environment extents.
temp_model_for_dims = AudioMesaModel(
    room_config=default_room.generate(),
    source_folders_dict=default_source_folders,
    env_config_overrides={
        "corners": default_room.corners,
        "num_sources": len(default_source_folders)
    }
)

# These parameters will be used by Mesa to instantiate the model
# and are also used here to set up the canvas dimensions.
model_params = {
    "room_config": default_room.generate(),
    "source_folders_dict": default_source_folders,
    "env_config_overrides": {
        "corners": default_room.corners,
        "max_order": 3, # Example: can be different from temp_model if needed for actual sim
        "num_sources": len(default_source_folders)
    },
    # Dimensions for canvas setup, derived from temp_model's space
    # This ensures the canvas accurately reflects the model's coordinate system.
    "canvas_width_meters": temp_model_for_dims.space.x_max - temp_model_for_dims.space.x_min,
    "canvas_height_meters": temp_model_for_dims.space.y_max - temp_model_for_dims.space.y_min,
}

canvas_element = ContinuousCanvas(
    agent_portrayal,
    model_params["canvas_width_meters"], 
    model_params["canvas_height_meters"],
    CANVAS_WIDTH_PX, # Pixel width
    CANVAS_HEIGHT_PX  # Pixel height
)

server = ModularServer(
    AudioMesaModel,
    [canvas_element],
    "Audio Navigation Environment",
    model_params # Pass the whole dict, Mesa will pick out what AudioMesaModel's __init__ needs
)
server.port = 8521 # Default Mesa port

if __name__ == "__main__":
    # The sys.path modifications are now at the top of the script.
    # This ensures all imports are resolved before any other code runs.
    print(f"Attempting to launch server on port {server.port}")
    print(f"Model parameters being used: {model_params}")
    print("If the server fails to start, check console for import errors, especially for 'room_types' or 'AudioEnv' dependencies.")
    print("Ensure sound paths in 'default_source_folders' are correct relative to the project root.")
    server.launch()
