from video_mode.example_video_mode_random import get_video_mode_component
from voltage_control.example_voltage_control import get_voltage_control_component
from qua_dashboards.core import build_dashboard
from qua_dashboards.utils import setup_logging


def main():
    logger = setup_logging(__name__)
    logger.info("Starting Combined Video Mode + Voltage Control dashboard")

    voltage_control_component = get_voltage_control_component()
    voltage_control_component.layout_columns = 3

    video_mode_component = get_video_mode_component()
    video_mode_component.layout_columns = 8

    # Connect the voltage control parameters to the video mode component
    x_axis = video_mode_component.data_acquirer.x_axis
    y_axis = video_mode_component.data_acquirer.y_axis
    x_axis.offset_parameter = voltage_control_component.voltage_parameters[0]
    y_axis.offset_parameter = voltage_control_component.voltage_parameters[1]

    app = build_dashboard(
        components=[video_mode_component, voltage_control_component],
        title="Combined Video Mode + Voltage Control Dashboard",
    )
    logger.info("Dashboard built. Access at: http://127.0.0.1:8050/")
    app.run(debug=True, port=8050, use_reloader=False, host="0.0.0.0")


if __name__ == "__main__":
    main()
