from qua_dashboards.video_mode import *

if __name__ == "__main__":
    import logging

    # Update the logging configuration
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    x_axis = SweepAxis(name="X", span=0.1, points=51)
    y_axis = SweepAxis(name="Y", span=0.1, points=101)

    data_acquirer = RandomDataAcquirer(
        x_axis=x_axis,
        y_axis=y_axis,
        num_averages=5,
        acquire_time=0.1,
    )

    live_plotter = VideoModeApp(data_acquirer=data_acquirer, update_interval=0.1)
    live_plotter.run()
