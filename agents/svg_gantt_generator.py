#!/usr/bin/env python3
"""
SVG Gantt Chart Generator
=========================

A comprehensive Python class for generating SVG Gantt charts with seconds-based timing.
No external dependencies required beyond Python standard library.

Author: Custom Implementation
License: MIT
"""

import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# (Task name, Bar End spec), Bar end: start, end, None (the same time stamp as another task spec)
TaskEndSpec = tuple[str, str | float | None]

@dataclass
class Task:
    """Represents a single task in the Gantt chart."""
    name: str
    start_time: float  # in seconds
    duration: float    # in seconds
    color: str = "#4A90E2"
    priority: str = "medium"
    resource: str = "default"
    description: str = ""
    
    @property
    def end_time(self) -> float:
        """Calculate end time of the task."""
        return self.start_time + self.duration


@dataclass
class Resource:
    """Represents a resource that can be assigned to tasks."""
    name: str
    color: str = "#6B7280"
    capacity: int = 1


class SVGGanttGenerator:
    """
    A comprehensive SVG Gantt chart generator that handles seconds-based timing.
    
    Features:
    - Seconds-based timing with sub-second precision
    - Multiple resources and priority levels
    - Customizable colors and styling
    - Dependencies and milestones
    - Grid lines and time markers
    - Legends and annotations
    """
    
    def __init__(self, 
                 width: int = 900,
                 height: int = 600,
                 margin_left: int = 150,
                 margin_top: int = 80,
                 margin_right: int = 50,
                 margin_bottom: int = 50,
                 row_height: int = 40,
                 font_family: str = "Arial, sans-serif"):
        """
        Initialize the Gantt chart generator.
        
        Args:
            width: Total SVG width
            height: Total SVG height
            margin_left: Left margin for task labels
            margin_top: Top margin for title and time axis
            margin_right: Right margin
            margin_bottom: Bottom margin
            row_height: Height of each task row
            font_family: Font family for text elements
        """
        self.width = width
        self.height = height
        self.margin_left = margin_left
        self.margin_top = margin_top
        self.margin_right = margin_right
        self.margin_bottom = margin_bottom
        self.row_height = row_height
        self.font_family = font_family
        
        self.tasks: List[Task] = []
        self.resources: Dict[str, Resource] = {}
        self.dependencies: List[Tuple[TaskEndSpec, TaskEndSpec]] = []  # (from_task, to_task)
        self.milestones: List[Tuple[float, str]] = []  # (time, label)
        
        # Chart dimensions
        self.chart_width = width - margin_left - margin_right
        self.chart_height = height - margin_top - margin_bottom
        
        # Timing
        self.min_time = 0
        self.max_time = 100
        self.time_scale = 1.0
        
        # Styling
        self.colors = {
            'critical': '#d63031',
            'high': '#e17055',
            'medium': '#fdcb6e',
            'low': '#81ecec',
            'background': '#ffffff',
            'grid': '#e5e5e5',
            'text': '#2d3436',
            'axis': '#636e72'
        }
        
        # Title and description
        self.title = "Gantt Chart"
        self.description = ""
    
    def add_task(self, task: Task) -> None:
        """Add a task to the Gantt chart."""
        self.tasks.append(task)
        self._update_time_bounds()
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the chart."""
        self.resources[resource.name] = resource

    def add_dependency(self, from_task: TaskEndSpec, to_task: TaskEndSpec) -> None:
        """Add a dependency between two tasks."""
        self.dependencies.append((from_task, to_task))
    
    def add_milestone(self, time: float, label: str) -> None:
        """Add a milestone marker at the specified time."""
        self.milestones.append((time, label))
    
    def set_title(self, title: str, description: str = "") -> None:
        """Set the chart title and description."""
        self.title = title
        self.description = description
    
    def _update_time_bounds(self) -> None:
        """Update the time bounds based on current tasks."""
        if not self.tasks:
            return
            
        self.min_time = min(task.start_time for task in self.tasks)
        self.max_time = max(task.end_time for task in self.tasks)
        
        # Add some padding
        time_range = self.max_time - self.min_time
        padding = time_range * 0.05  # 5% padding
        self.min_time = max(0, self.min_time - padding)
        self.max_time += padding
        
        # Update time scale
        self.time_scale = self.chart_width / (self.max_time - self.min_time)
    
    def _time_to_x(self, time: float) -> float:
        """Convert time to X coordinate."""
        return self.margin_left + (time - self.min_time) * self.time_scale
    
    def _generate_css(self) -> str:
        """Generate CSS styles for the SVG."""
        return f"""
        <style>
            .gantt-title {{ 
                font-family: {self.font_family}; 
                font-size: 18px; 
                font-weight: bold; 
                fill: {self.colors['text']};
            }}
            .gantt-description {{ 
                font-family: {self.font_family}; 
                font-size: 12px; 
                fill: {self.colors['text']};
            }}
            .task-label {{ 
                font-family: {self.font_family}; 
                font-size: 12px; 
                fill: {self.colors['text']};
            }}
            .time-label {{ 
                font-family: {self.font_family}; 
                font-size: 10px; 
                fill: {self.colors['text']};
            }}
            .grid-line {{ 
                stroke: {self.colors['grid']}; 
                stroke-width: 1; 
            }}
            .axis-line {{ 
                stroke: {self.colors['axis']}; 
                stroke-width: 2; 
            }}
            .task-bar {{ 
                stroke: #333; 
                stroke-width: 1; 
            }}
            .task-text {{ 
                font-family: {self.font_family}; 
                font-size: 10px; 
                font-weight: bold;
            }}
            .dependency-line {{ 
                stroke: #999; 
                stroke-width: 2; 
                stroke-dasharray: 3,3; 
                fill: none;
            }}
            .milestone-line {{ 
                stroke: #e74c3c; 
                stroke-width: 3; 
            }}
            .milestone-text {{ 
                font-family: {self.font_family}; 
                font-size: 10px; 
                fill: #e74c3c; 
                font-weight: bold;
            }}
            .priority-critical {{ fill: {self.colors['critical']}; }}
            .priority-high {{ fill: {self.colors['high']}; }}
            .priority-medium {{ fill: {self.colors['medium']}; }}
            .priority-low {{ fill: {self.colors['low']}; }}
        </style>
        """
    
    def _generate_time_axis(self) -> str:
        """Generate the time axis with grid lines and labels."""
        svg_parts = []
        
        # Main axis line
        svg_parts.append(f'<line x1="{self.margin_left}" y1="{self.margin_top}" '
                        f'x2="{self.width - self.margin_right}" y2="{self.margin_top}" '
                        f'class="axis-line"/>')
        
        # Calculate time intervals
        time_range = self.max_time - self.min_time
        
        # Determine appropriate interval
        if time_range <= 10:
            interval = 1
        elif time_range <= 60:
            interval = 5
        elif time_range <= 300:
            interval = 30
        elif time_range <= 600:
            interval = 60
        else:
            interval = math.ceil(time_range / 10)
        
        # Generate time markers
        current_time = math.ceil(self.min_time / interval) * interval
        chart_bottom = self.margin_top + len(self.tasks) * self.row_height
        
        while current_time <= self.max_time:
            x = self._time_to_x(current_time)
            
            # Grid line
            svg_parts.append(f'<line x1="{x}" y1="{self.margin_top}" '
                           f'x2="{x}" y2="{chart_bottom}" class="grid-line"/>')
            
            # Time label
            time_str = self._format_time(current_time)
            svg_parts.append(f'<text x="{x}" y="{self.margin_top - 10}" '
                           f'class="time-label" text-anchor="middle">{time_str}</text>')
            
            current_time += interval
        
        return '\n'.join(svg_parts)
    
    def _format_time(self, time: float) -> str:
        """Format time for display."""
        if time < 60:
            return f"{time:.1f}s".rstrip('0').rstrip('.')
        elif time < 3600:
            minutes = int(time // 60)
            seconds = time % 60
            if seconds == 0:
                return f"{minutes}m"
            else:
                return f"{minutes}m{seconds:.0f}s"
        else:
            hours = int(time // 3600)
            minutes = int((time % 3600) // 60)
            if minutes == 0:
                return f"{hours}h"
            else:
                return f"{hours}h{minutes}m"
    
    def _generate_tasks(self) -> str:
        """Generate SVG for all tasks."""
        svg_parts = []
        
        for i, task in enumerate(self.tasks):
            y = self.margin_top + i * self.row_height + 10
            
            # Task label
            svg_parts.append(f'<text x="5" y="{y + 15}" class="task-label">{task.name}</text>')
            
            # Task bar
            x = self._time_to_x(task.start_time)
            width = max(1, (task.end_time - task.start_time) * self.time_scale)
            
            # Determine color based on priority or custom color
            if task.color.startswith('#'):
                fill_color = task.color
            else:
                fill_color = self.colors.get(task.priority, task.color)
            
            svg_parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="25" '
                           f'fill="{fill_color}" class="task-bar" opacity="0.8"/>')
            
            # Task duration text (if bar is wide enough)
            if width > 40:
                text_color = self._get_contrast_color(fill_color)
                duration_text = self._format_time(task.duration)
                svg_parts.append(f'<text x="{x + width/2}" y="{y + 17}" '
                               f'class="task-text" text-anchor="middle" '
                               f'fill="{text_color}">{duration_text}</text>')
            
            # Priority badge (if not using custom color)
            if not task.color.startswith('#') and task.priority != 'medium':
                badge_x = x + 5
                svg_parts.append(f'<text x="{badge_x}" y="{y + 12}" '
                               f'class="task-text" fill="white" font-size="8px">'
                               f'{task.priority.upper()}</text>')

            # task.description
            if task.description:
                desc_x = x + width + 5
                svg_parts.append(f'<text x="{desc_x}" y="{y + 15}" '
                               f'class="task-text" fill="#666" font-size="10px">'
                               f'{task.description}</text>')
        
        return '\n'.join(svg_parts)
    
    def _get_contrast_color(self, hex_color: str) -> str:
        """Get contrasting color (black or white) for given hex color."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        return 'white' if luminance < 0.5 else 'black'
    
    def _generate_dependencies(self) -> str:
        """Generate dependency arrows between tasks."""
        if not self.dependencies:
            return ""
        
        svg_parts = []
        
        # Create arrowhead marker
        svg_parts.append('''
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
        </defs>''')
        
        # Task name to index mapping
        task_map = {task.name: i for i, task in enumerate(self.tasks)}
        
        for (from_task, from_end), (to_task, to_end) in self.dependencies:
            if from_end is None and to_end is None:
                print(f"Warning: Both ends of dependency from '{from_task}' to '{to_task}' are None. Skipping.")
                continue

            if from_task in task_map and to_task in task_map:
                from_idx = task_map[from_task]
                to_idx = task_map[to_task]
                
                from_task_obj = self.tasks[from_idx]
                to_task_obj = self.tasks[to_idx]
                
                # Calculate positions
                from_time = (
                    from_end if isinstance(from_end, float) else
                    from_task_obj.end_time if from_end == "end" else
                    from_task_obj.start_time if from_end == "start" else None
                    )
                to_time = (
                    to_end if isinstance(to_end, float) else
                    to_task_obj.start_time if to_end == "start" else
                    to_task_obj.end_time if to_end == "end" else None
                    )
                if from_time is None:
                    from_time = to_time
                if to_time is None:
                    to_time = from_time
                if from_time is None or to_time is None:
                    raise ValueError(f"Invalid dependency times for '{from_task}' to '{to_task}'")
                from_x = self._time_to_x(from_time)
                from_y = self.margin_top + from_idx * self.row_height + 22
                
                to_x = self._time_to_x(to_time)
                to_y = self.margin_top + to_idx * self.row_height + 22
                
                # Draw dependency line
                svg_parts.append(f'<line x1="{from_x}" y1="{from_y}" '
                               f'x2="{to_x}" y2="{to_y}" '
                               f'class="dependency-line" marker-end="url(#arrowhead)"/>')
        
        return '\n'.join(svg_parts)
    
    def _generate_milestones(self) -> str:
        """Generate milestone markers."""
        if not self.milestones:
            return ""
        
        svg_parts = []
        chart_bottom = self.margin_top + len(self.tasks) * self.row_height
        
        for time, label in self.milestones:
            x = self._time_to_x(time)
            
            # Milestone line
            svg_parts.append(f'<line x1="{x}" y1="{self.margin_top}" '
                           f'x2="{x}" y2="{chart_bottom}" class="milestone-line"/>')
            
            # Milestone label
            svg_parts.append(f'<text x="{x + 5}" y="{self.margin_top + 15}" '
                           f'class="milestone-text">{label}</text>')
        
        return '\n'.join(svg_parts)
    
    def _generate_legend(self) -> str:
        """Generate a legend for priorities and resources."""
        if not self.resources and all(task.priority == 'medium' for task in self.tasks):
            return ""
        
        svg_parts = []
        legend_y = self.height - 40
        
        # Priority legend
        priorities = set(task.priority for task in self.tasks)
        if len(priorities) > 1:
            svg_parts.append(f'<text x="20" y="{legend_y}" class="task-label">Priority:</text>')
            x_offset = 80
            
            for priority in sorted(priorities):
                color = self.colors.get(priority, '#999')
                svg_parts.append(f'<rect x="{x_offset}" y="{legend_y - 12}" '
                               f'width="15" height="12" fill="{color}"/>')
                svg_parts.append(f'<text x="{x_offset + 20}" y="{legend_y - 2}" '
                               f'class="time-label">{priority.title()}</text>')
                x_offset += 80
        
        return '\n'.join(svg_parts)
    
    def generate_svg(self) -> str:
        """Generate the complete SVG Gantt chart."""
        if not self.tasks:
            raise ValueError("No tasks added to the Gantt chart")
        
        self._update_time_bounds()
        
        # Calculate required height based on tasks
        required_height = self.margin_top + len(self.tasks) * self.row_height + self.margin_bottom
        if required_height > self.height:
            self.height = required_height
        
        svg_parts = [
            f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="white"/>',  # Set background to white
            self._generate_css(),
            ''
        ]
        
        # Title and description
        svg_parts.append(f'<text x="{self.width/2}" y="25" class="gantt-title" '
                        f'text-anchor="middle">{self.title}</text>')
        
        if self.description:
            svg_parts.append(f'<text x="{self.width/2}" y="45" class="gantt-description" '
                           f'text-anchor="middle">{self.description}</text>')
        
        # Chart components
        svg_parts.extend([
            self._generate_time_axis(),
            self._generate_tasks(),
            self._generate_dependencies(),
            self._generate_milestones(),
            self._generate_legend(),
            '</svg>'
        ])
        
        return '\n'.join(svg_parts)
    
    def save_svg(self, filename: str) -> None:
        """Save the SVG to a file."""
        svg_content = self.generate_svg()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"Gantt chart saved to {filename}")
    
    def get_svg_string(self) -> str:
        """Return the SVG content as a string without saving to file."""
        return self.generate_svg()
    
    def save_html(self, filename: str, include_styling: bool = True) -> None:
        """Save as an HTML file with optional enhanced styling."""
        svg_content = self.generate_svg()
        
        html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {"" if not include_styling else """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .gantt-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 100%;
            overflow-x: auto;
        }
        .info {
            margin-bottom: 15px;
            color: #666;
            font-size: 14px;
        }
    </style>"""}
</head>
<body>
    {"" if not include_styling else f'<div class="gantt-container">'}
    {"" if not include_styling else f'<div class="info"><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>'}
    {svg_content}
    {"" if not include_styling else '</div>'}
</body>
</html>'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"HTML Gantt chart saved to {filename}")


if __name__ == "__main__":

    # Load the JSON file
    with open("graph_20250627_160252.json", "r") as file:
        data = json.load(file)

    from fork_manager import ForkManager

    gantt = ForkManager.generate_gantt_chart(data)

    # Save the Gantt chart as an SVG file
    gantt.save_svg("gantt_chart.svg")

    pass
