import threading
import time
from datetime import datetime
import psutil

def monitor_system(interval=1.0, file_path="memory_usage_log.txt"):
    """Monitors memory usage and writes data to a file."""
    with open(file_path, "a") as file:  # Open file in append mode
        while True:
            # Get memory usage
            mem = psutil.virtual_memory().percent
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            output = f"{timestamp} - Memory: {mem}%\n"
            
            print(output, end='')  # Print to console
            file.write(output)  # Write to file
            file.flush()  # Ensure data is written to disk
            
            # Sleep for the specified interval
            time.sleep(interval)

def start_monitoring(output_file_path, interval=1.0):
    """Starts the system monitoring in a separate thread."""
    threading.Thread(target=monitor_system, args=(interval, output_file_path), daemon=True).start()

if __name__ == "__main__":
    output_file_path = "memory_usage_log.txt"
    # Start the monitoring
    start_monitoring(output_file_path)
    input("Monitoring memory usage. Press Enter to stop...\n")
