"""
Dash component for displaying and controlling a series of voltage parameters.
"""

import logging
from typing import Dict, Sequence

import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
)

from qua_dashboards.core import BaseComponent, ParameterProtocol

from .voltage_control_row import VoltageControlRow, format_voltage

logger = logging.getLogger(__name__)

DEFAULT_INPUT_CLASS_NAME = ""  # Input fields will always have this class
COMPONENT_MAX_WIDTH = "450px"


class VoltageControlComponent(BaseComponent):
    """
    A Dash component to display and set voltages from multiple sources.
    Uses VoltageControlRow to manage individual parameter rows.
    Simplified: No edit highlighting, no auto-defocus on enter.
    """

    def __init__(
        self,
        component_id: str,
        voltage_parameters: Sequence[ParameterProtocol],
        update_interval_ms: int = 1000,
        layout_columns: int = 3,
        step_size: float = 100e-6,
    ):
        super().__init__(component_id=component_id)
        self.voltage_parameters = voltage_parameters
        self.update_interval_ms = update_interval_ms

        self._initial_values_loaded = False  # To ensure first update populates values

        self._row_components: Dict[str, VoltageControlRow] = {}
        input_id_type = self._get_id_type_str("input")
        for param in self.voltage_parameters:
            row = VoltageControlRow(input_id_type=input_id_type, param=param)
            self._row_components[param.name] = row

        self.layout_columns = layout_columns
        self.step_size = step_size

    def _get_id_type_str(self, element_name: str) -> str:
        return f"comp-{self.component_id}-{element_name}"

    def get_layout(self) -> html.Div:
        if not self._row_components:
            return html.Div(
                "No voltage parameters configured.",
                style={"maxWidth": COMPONENT_MAX_WIDTH},
            )
        input_ids = [
            {"type": self._get_id_type_str("input"), "index": param.name} 
            for param in self.voltage_parameters
        ]
        return html.Div(
            [
                dcc.Interval(
                    id=self._get_id("update-interval"),
                    interval=self.update_interval_ms,
                    n_intervals=0,
                ),
                dcc.Store(
                    id=self._get_id("keyboard-config"),
                    data={
                        "input_ids": input_ids,
                        "step_size": self.step_size,
                    }
                ),
                dcc.Store(id=self._get_id("keyboard-dummy")),  # Changed from html.Div
                *[row.get_layout() for row in self._row_components.values()],
            ],
            style={"maxWidth": COMPONENT_MAX_WIDTH, "margin": "0 auto"},
        )
    
    def register_callbacks(self, app: dash.Dash) -> None:
        if not self.voltage_parameters:
            return

        output_list_for_periodic_update = []
        input_id_type_str = self._get_id_type_str("input")
        for param in self.voltage_parameters:
            param_input_id = {"type": input_id_type_str, "index": param.name}
            output_list_for_periodic_update.append(
                Output(param_input_id, "value", allow_duplicate=True)
            )
            output_list_for_periodic_update.append(
                Output(
                    param_input_id, "className", allow_duplicate=True
                )  # Keep for structure if row outputs it
            )

        if not output_list_for_periodic_update:
            return

        @app.callback(
            output_list_for_periodic_update,
            Input(self._get_id("update-interval"), "n_intervals"),
            prevent_initial_call=True,
        )
        def periodic_update(_n_intervals: int):
            outputs_tuple_elements = []

            for param in self.voltage_parameters:
                param_name = param.name
                control_row = self._row_components[param_name]

                try:
                    live_value = param.get_latest()
                    live_text = format_voltage(live_value)
                except Exception as e:
                    logger.error(f"Err get_latest for {param_name}: {e}", exc_info=True)
                    live_text = control_row.current_input_text
                if control_row.current_input_text != control_row.last_committed_text:
                    outputs_tuple_elements.append(dash.no_update)
                else:
                    if live_text != control_row.current_input_text:
                        control_row.current_input_text = live_text
                        control_row.last_committed_text = live_text
                        outputs_tuple_elements.append(live_text)
                    else:
                        outputs_tuple_elements.append(dash.no_update)
                outputs_tuple_elements.append(DEFAULT_INPUT_CLASS_NAME)

            return tuple(outputs_tuple_elements)


        for row_component in self._row_components.values():
            row_component.register_callbacks(app)

        app.clientside_callback(
            r"""
            function(config) {
                // If nothing configured yet, don't do anything
                if (!config || !config.input_ids) {
                    return null;
                }

                // Only attach once per page load
                if (window._voltageKeyboardListenerAttached) {
                    return null;
                }
                window._voltageKeyboardListenerAttached = true;

                var inputIds = config.input_ids || [];
                var stepSize = config.step_size || 0.00001;

                function findVoltageInput(index) {
                    if (index < 0 || index >= inputIds.length) {
                        return null;
                    }
                    var id = inputIds[index];
                    var allInputs = document.querySelectorAll('input');
                    for (var i = 0; i < allInputs.length; i++) {
                        var el = allInputs[i];
                        var rawId = el.id;
                        if (!rawId) continue;
                        try {
                            var parsed = JSON.parse(rawId);
                            if (parsed && parsed.type === id.type && parsed.index === id.index) {
                                return el;
                            }
                        } catch (e) {
                            // Ignore non-JSON ids
                        }
                    }
                    return null;
                }

                function isVoltageInputElement(el) {
                    if (!el || !el.id) return false;
                    try {
                        var parsed = JSON.parse(el.id);
                        for (var i = 0; i < inputIds.length; i++) {
                            var id = inputIds[i];
                            if (id.type === parsed.type && id.index === parsed.index) {
                                return true;
                            }
                        }
                        return false;
                    } catch (e) {
                        return false;
                    }
                }

                document.addEventListener('keydown', function(e) {
                    var activeEl = document.activeElement;
                    var tagName = activeEl ? activeEl.tagName.toLowerCase() : '';
                    var isInInput = (tagName === 'input' || tagName === 'textarea');
                    var isInVoltageInput = isInInput && isVoltageInputElement(activeEl);

                    // QWERTY hotkeys: q-p, a-l, z-m
                    var shortcuts = [
                        'q','w','e','r','t','y','u','i','o','p',
                        'a','s','d','f','g','h','j','k','l',
                        'z','x','c','v','b','n','m'
                    ];
                    var key = (e.key || '').toLowerCase();
                    var idx = shortcuts.indexOf(key);

                    // Focus nth voltage input
                    if (idx !== -1 && (!isInInput || isInVoltageInput)) {
                        var targetEl = findVoltageInput(idx);
                        if (targetEl) {
                            e.preventDefault();
                            if (activeEl && typeof activeEl.blur === 'function') {
                                activeEl.blur();
                            }
                            targetEl.focus();
                            if (typeof targetEl.select === 'function') {
                                targetEl.select();
                            }
                        }
                    }

                    // Arrow up/down: increment/decrement current voltage input
                    if (isInVoltageInput && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                        e.preventDefault();

                        var currentVal = parseFloat(activeEl.value);
                        if (!isFinite(currentVal)) {
                            currentVal = 0;
                        }

                        var delta = (e.key === 'ArrowUp') ? stepSize : -stepSize;
                        if (e.shiftKey) delta *= 10;
                        if (e.ctrlKey)  delta *= 100;

                        var newVal = currentVal + delta;
                        var newValStr = newVal.toFixed(9).replace(/\.?0+$/, '');

                        try {
                            var desc = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype,
                                'value'
                            );
                            if (desc && typeof desc.set === 'function') {
                                desc.set.call(activeEl, newValStr);
                            } else {
                                activeEl.value = newValStr;
                            }
                        } catch (err) {
                            activeEl.value = newValStr;
                        }

                        activeEl.dispatchEvent(new Event('input', { bubbles: true }));
                    }

                    // Do not intercept Enter – Dash needs it for n_submit
                });

                // We don't actually use the store value; null is fine
                return null;
            }
            """,
            Output(self._get_id("keyboard-dummy"), "data"),
            Input(self._get_id("keyboard-config"), "data"),
            prevent_initial_call=False,
        )
