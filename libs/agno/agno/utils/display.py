import json
from datetime import datetime
from typing import Optional

import yaml
from rich import print
from rich.console import Console
from rich.pretty import Pretty


def custom_serializer(obj):
    """
    Custom serializer for non-serializable objects.
    Converts objects with a __dict__ attribute to their dictionary representation.
    """
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")  # Format datetime objects
    return str(obj)  # Fallback: Convert the object to a string


def filter_instance_dict(instance):
    """
    Recursively filter the __dict__ of a class instance, ignoring empty values.
    If an element is a list, apply the filter to each element recursively.

    Args:
        instance: The class instance or value to be filtered.

    Returns:
        dict or list or value: A filtered dictionary, list, or value with non-empty elements.
    """
    if isinstance(instance, (list, tuple)):
        # Recursively filter each element in the list
        filtered_list = [
            filter_instance_dict(item)
            for item in instance
            if item not in (None, 0, "", [], {}, set(), ())  # Ignore empty values
        ]
        # Remove the list if all elements are empty
        return filtered_list if any(filtered_list) else None

    if hasattr(instance, "__dict__"):
        if isinstance(instance.__dict__, dict):
            return filter_instance_dict(instance.__dict__)
        elif callable(instance.__dict__):
            return filter_instance_dict(instance.__dict__())

    if isinstance(instance, dict):
        # Recursively filter the __dict__ of the instance
        filtered_dict = {}
        for key, value in instance.items():
            if key.startswith("_") or "api_key" in key:
                continue
            if value in (None, 0, "", [], {}, set(), ()):
                continue
            sub = filter_instance_dict(value)
            if sub in (None, 0, "", [], {}, set(), ()):
                continue
            if hasattr(value, "__class__") and value.__class__.__module__ != "builtins":
                key = f"{key} <{value.__class__.__module__}.{value.__class__.__name__}>"
            filtered_dict[key] = sub
        # Remove the dictionary if all elements are empty
        return filtered_dict if any(filtered_dict.values()) else None

    # Return the value as is if it's not a list or class instance
    return instance


def display(instance, to_file: Optional[str] = None):
    """
    Display the __dict__ of a class instance using rich.pprint, ignoring empty values.

    Args:
        instance: The class instance whose __dict__ is to be displayed.
        to_file: Optional file path to save the filtered dictionary in JSON format.
    """
    if not hasattr(instance, "__dict__"):
        raise ValueError("The provided object does not have a __dict__ attribute.")

    # Recursively filter the instance's __dict__
    filtered_dict = filter_instance_dict(instance)

    title = f"<{instance.__class__.__module__}.{instance.__class__.__name__}>"
    if to_file:
        ext = to_file.lower().split(".")[-1]
        if ext == "json":
            # Write the filtered dictionary to a file in JSON format
            with open(to_file, "w", encoding="utf-8") as f:
                json.dump(
                    obj=filtered_dict,
                    fp=f,
                    indent=2,
                    ensure_ascii=False,
                    default=custom_serializer,
                )
        elif ext in [".yaml", "yml"]:
            # Write the filtered dictionary to a file in YAML format
            with open(to_file, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n")
                yaml.dump(
                    data=filtered_dict,
                    stream=f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
        else:
            raise ValueError("Unsupported file format. Use 'json' or 'yml'.")
    else:
        # Display class name and filtered dictionary
        console = Console()
        print(f"[bold cyan]{title}[/bold cyan]")
        console.print(Pretty(filtered_dict, indent_size=2))
