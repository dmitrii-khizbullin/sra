from typing import Any

from agents.svg_gantt_generator import SVGGanttGenerator, Task


def generate_gantt_chart(records_json: dict[str, dict[str, Any]]) -> SVGGanttGenerator:
    gantt = SVGGanttGenerator()
    gantt.set_title("Thread Execution Gantt Chart", "Visualization of thread execution timelines.")

    colormap = [
        "#ff6b6b", "#4ecdc4", "#ffa500", "#8a2be2", "#00ff7f",
    ]

    time_zero = None
    for tid, record in records_json.items():
        fork_time = record["fork_time"]
        if time_zero is None or fork_time < time_zero:
            time_zero = fork_time
    if time_zero is not None:
        for tid, record in records_json.items():
            start_time = record["start_time"]
            if start_time is None:
                start_time = record["fork_time"]
                end_time = record["fork_time"] + 0.1 # a bug here
            end_time = record["end_time"]
            if end_time is None:
                duration = 1.0  # Default duration if end_time is not set
            else:
                duration = end_time - start_time
            relative_start_time = start_time - time_zero
            color = colormap[record["level"] % len(colormap)]
            gantt.add_task(Task(name=f"Thread {tid}",
                                start_time=relative_start_time,
                                duration=duration,
                                color=color,
                                description=f"Level {record['level']}",
                                ))
    
    # add dependencies
    for tid, record in records_json.items():
        for child_tid in record['child_tids']:
            child_record = records_json[child_tid]
            fork_time = child_record['fork_time'] - time_zero
            if child_record['start_time'] is None:
                gantt.add_dependency((f"Thread {tid}", fork_time), (f"Thread {child_tid}", fork_time))
                continue
            start_time = child_record['start_time'] - time_zero
            gantt.add_dependency((f"Thread {tid}", fork_time), (f"Thread {child_tid}", start_time))
            if child_record['end_time'] is None or child_record['join_time'] is None:
                continue
            end_time = child_record['end_time'] - time_zero
            join_time = child_record['join_time'] - time_zero
            gantt.add_dependency((f"Thread {child_tid}", end_time), (f"Thread {tid}", join_time))

    return gantt
