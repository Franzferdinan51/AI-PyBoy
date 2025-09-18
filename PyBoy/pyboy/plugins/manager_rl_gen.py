#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#
"""
RL Plugin Manager Generator

This script generates the manager.py file with RL-specific plugin support.
Run this script to update the plugin manager with RL wrappers.
"""

import re
import os

# RL Game Wrappers (in addition to existing ones)
rl_game_wrappers = [
    "RLGameWrapperSuperMarioLand",
    "RLGameWrapperTetris",
    "RLGameWrapperPokemonGen1",
    "RLGameWrapperGeneral"
]

# All game wrappers (existing + RL)
all_game_wrappers = [
    "GameWrapperSuperMarioLand",
    "GameWrapperTetris",
    "GameWrapperKirbyDreamLand",
    "GameWrapperPokemonGen1",
    "GameWrapperPokemonPinball",
] + rl_game_wrappers

# Windows and plugins remain the same
windows = ["WindowSDL2", "WindowOpenGL", "WindowNull", "Debug"]
plugins = [
    "AutoPause",
    "RecordReplay",
    "Rewind",
    "ScreenRecorder",
    "ScreenshotRecorder",
    "DebugPrompt",
] + all_game_wrappers

all_plugins = windows + plugins


def to_snake_case(s):
    """Convert CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def skip_lines(iterator, stop):
    """Skip lines in iterator until finding stop marker."""
    while True:
        if next(iterator).strip().startswith(stop):
            break


def generate_manager_py():
    """Generate manager.py with RL plugin support."""
    out_lines = []

    with open("manager.py", "r") as f:
        line_iter = iter(f.readlines())
        while True:
            line = next(line_iter, None)
            if line is None:
                break

            # Find place to inject
            if line.strip().startswith("# foreach"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# foreach")

                skip_lines(line_iter, "# foreach end")

                _, foreach, plugin_type, fun = line.strip().split(" ", 3)
                for p in eval(plugin_type):
                    p_name = to_snake_case(p)
                    lines.append(f"if self.{p_name}_enabled:\n")
                    for sub_fun in fun.split(", "):
                        sub_fun = sub_fun.replace("[]", f"self.{p_name}")
                        lines.append(f"    {sub_fun}\n")

                lines.append("# foreach end\n")
                out_lines.extend([indentation + l for l in lines])

            elif line.strip().startswith("# gamewrapper"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# gamewrapper")

                skip_lines(line_iter, "# gamewrapper end")

                # Modified gamewrapper logic - prioritize RL wrappers
                for p in rl_game_wrappers + [gw for gw in all_game_wrappers if gw not in rl_game_wrappers]:
                    p_name = to_snake_case(p)
                    lines.append(f"if self.{p_name}_enabled: return self.{p_name}\n")

                lines.append("# gamewrapper end\n")
                out_lines.extend([indentation + l for l in lines])

            elif line.strip().startswith("# plugins_enabled"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# plugins_enabled")

                skip_lines(line_iter, "# plugins_enabled end")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"self.{p_name} = {p}(pyboy, mb, pyboy_argv)\n")
                    lines.append(f"self.{p_name}_enabled = self.{p_name}.enabled()\n")

                lines.append("# plugins_enabled end\n")
                out_lines.extend([indentation + l for l in lines])

            elif line.strip().startswith("# yield_plugins"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# yield_plugins")

                skip_lines(line_iter, "# yield_plugins end")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"yield {p}.argv\n")

                lines.append("# yield_plugins end\n")
                out_lines.extend([indentation + l for l in lines])

            elif line.strip().startswith("# imports"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# imports")

                skip_lines(line_iter, "# imports end")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"from pyboy.plugins.{p_name} import {p} # noqa\n")

                lines.append("# imports end\n")
                out_lines.extend([indentation + l for l in lines])

            else:
                out_lines.append(line)

    return out_lines


def generate_manager_pxd():
    """Generate manager.pxd with RL plugin support."""
    out_lines = []

    with open("manager.pxd", "r") as f:
        line_iter = iter(f.readlines())
        while True:
            line = next(line_iter, None)
            if line is None:
                break

            if line.strip().startswith("# plugin_cdef"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# plugin_cdef")

                skip_lines(line_iter, "# plugin_cdef end")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"cdef public {p} {p_name}\n")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"cdef bint {p_name}_enabled\n")

                lines.append("# plugin_cdef end\n")
                out_lines.extend([indentation + l for l in lines])

            elif line.strip().startswith("# imports"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# imports")

                skip_lines(line_iter, "# imports end")

                for p in all_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f"from pyboy.plugins.{p_name} cimport {p}\n")

                lines.append("# imports end\n")
                out_lines.extend([indentation + l for l in lines])

            else:
                out_lines.append(line)

    return out_lines


def generate_init_py():
    """Generate __init__.py with RL plugin support."""
    out_lines = []

    with open("__init__.py", "r") as f:
        line_iter = iter(f.readlines())
        while True:
            line = next(line_iter, None)
            if line is None:
                break

            if line.strip().startswith("# docs exclude"):
                lines = [line.strip() + "\n"]
                indentation = " " * line.index("# docs exclude")

                skip_lines(line_iter, "# docs exclude end")

                # Exclude RL plugins from documentation by default
                excluded_plugins = sorted(list((set(all_plugins) - set(all_game_wrappers)) | set(["manager", "manager_gen"])))

                for p in excluded_plugins:
                    p_name = to_snake_case(p)
                    lines.append(f'"{p_name}": False,\n')

                lines.append("# docs exclude end\n")
                out_lines.extend([indentation + l for l in lines])

            else:
                out_lines.append(line)

    return out_lines


def backup_files():
    """Create backups of original files."""
    timestamp = re.sub(r'[^0-9]', '_', str(os.times()))

    for filename in ["manager.py", "manager.pxd", "__init__.py"]:
        if os.path.exists(filename):
            backup_name = f"{filename}.backup_{timestamp}"
            with open(filename, 'r') as src, open(backup_name, 'w') as dst:
                dst.write(src.read())
            print(f"Backed up {filename} to {backup_name}")


def main():
    """Main function to generate updated manager files."""
    print("Generating RL-enabled plugin manager...")

    # Create backups
    backup_files()

    # Generate updated files
    try:
        # manager.py
        manager_lines = generate_manager_py()
        with open("manager.py", "w") as f:
            f.writelines(manager_lines)
        print("Updated manager.py with RL plugin support")

        # manager.pxd
        pxd_lines = generate_manager_pxd()
        with open("manager.pxd", "w") as f:
            f.writelines(pxd_lines)
        print("Updated manager.pxd with RL plugin support")

        # __init__.py
        init_lines = generate_init_py()
        with open("__init__.py", "w") as f:
            f.writelines(init_lines)
        print("Updated __init__.py with RL plugin support")

        print("\nRL plugin manager generation completed successfully!")
        print(f"\nAdded RL Game Wrappers:")
        for wrapper in rl_game_wrappers:
            print(f"  - {wrapper}")

        print(f"\nPriority order (RL wrappers first):")
        for wrapper in rl_game_wrappers + [gw for gw in all_game_wrappers if gw not in rl_game_wrappers]:
            print(f"  - {wrapper}")

    except Exception as e:
        print(f"Error generating manager files: {e}")
        print("Please ensure you have the correct permissions and files exist.")


if __name__ == "__main__":
    main()