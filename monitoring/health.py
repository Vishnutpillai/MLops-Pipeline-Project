import psutil


def get_system_health():
    """
    Return current system resource usage.
    """

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": memory.percent,
        "disk_percent": disk.percent
    }


if __name__ == "__main__":
    health = get_system_health()

    print("System Health")
    print("--------------------")
    print(f"CPU Usage    : {health['cpu_percent']}%")
    print(f"Memory Usage : {health['memory_percent']}%")
    print(f"Disk Usage   : {health['disk_percent']}%")