from src.utils import load_json_to_dict, get_current_folder

# Load colors
cf = get_current_folder()
dark_violet = load_json_to_dict(f'{cf}/dark_violet.json')['hex']
darker_violet = load_json_to_dict(f'{cf}/darker_violet.json')['hex']
gray_violet = load_json_to_dict(f'{cf}/gray_violet.json')['hex']
light_violet = load_json_to_dict(f'{cf}/light_violet.json')['hex']
medium_violet = load_json_to_dict(f'{cf}/medium_violet.json')['hex']
light_medium_violet = load_json_to_dict(f'{cf}/light_medium_violet.json')['hex']
light_medium_violet_rgb = load_json_to_dict(f'{cf}/light_medium_violet.json')['rgb']
pink = load_json_to_dict(f'{cf}/pink.json')['hex']
light_lavander = load_json_to_dict(f'{cf}/light_lavander.json')['hex']
light_pink = load_json_to_dict(f'{cf}/light_pink.json')['hex']

pink_dataset = '#ea87e0'
dark_pink = '#d58bd4'

gate_color = darker_violet
pulsed_color = pink
mixed_color = light_lavander