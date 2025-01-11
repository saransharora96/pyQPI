import time
from datetime import timedelta
from rich.layout import Layout
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn


class CustomTimeRemainingColumn(TimeRemainingColumn):
    def render(self, task):
        """Override to calculate remaining time using task fields."""
        elapsed = task.fields.get("session_elapsed", 0)
        processed = task.fields.get("session_file_count", 0)
        if processed == 0:
            return "[cyan]--:--:--:--"
        files_remaining = task.total - task.completed
        time_per_file = elapsed / processed
        remaining_time = files_remaining * time_per_file

        # Convert remaining time to days, hours, minutes, and seconds
        td = timedelta(seconds=remaining_time)
        days, hours, minutes, seconds = td.days, td.seconds // 3600, (td.seconds // 60) % 60, td.seconds % 60
        return f"[cyan]{days}:{hours:02}:{minutes:02}:{seconds:02}"
    

def initialize_progress_bar(total_files, already_processed):
    """
    Initialize the progress bar and layout for file processing.

    Args:
        total_files (int): Total number of files to process.
        already_processed (int): Number of files already processed.

    Returns:
        Tuple[Progress, Layout, TaskID]: Initialized Progress instance, layout, and task ID.
    """
    progress = Progress(
        TextColumn("[bold blue]OVERALL PROGRESS:"),
        TextColumn(" | [cyan]Cell Line: {task.fields[current_cell_line]}"),
        TextColumn(" | [cyan]Dish: {task.fields[current_dish]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        CustomTimeRemainingColumn(),
        TextColumn("[green]{task.completed}/{task.total} files"),
        TextColumn(" | [green]{task.fields[seconds_per_file]:.2f} sec/file")
    )

    progress_layout = Layout()
    progress_layout.split(
        Layout(name="monitor", size=3),
        Layout(progress, name="progress", ratio=1)
    )

    progress_task = progress.add_task(
        "Processing Files",
        total=total_files,
        completed=already_processed,
        seconds_per_file=0.0,
        current_dish="Starting...",
        current_cell_line="Starting...",
        session_file_count=0,
        session_elapsed=0.0  # Initialize elapsed time for this session
    )

    return progress, progress_layout, progress_task

def update_progress_bar(progress, progress_task, session_start_time, session_file_count, result):
    """
    Update the progress bar with task completion information.

    Args:
        progress (Progress): The progress bar instance.
        progress_task (TaskID): The ID of the progress bar task.
        session_start_time (float): The start time of the session.
        session_file_count (int): Number of files processed in the session.
        result (dict): Result dictionary containing task details.
    """
    elapsed_time = time.time() - session_start_time
    seconds_per_file = elapsed_time / session_file_count if session_file_count > 0 else 0.0

    progress.update(
        progress_task,
        advance=1,
        seconds_per_file=seconds_per_file,
        current_dish=result.get("dish_number", "Unknown"),
        current_cell_line=result.get("radiation_resistance", "Unknown"),
        session_file_count=session_file_count,
        session_elapsed=elapsed_time
    )