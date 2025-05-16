// This file is used to handle the blur event for the input field after the user has finished typing.
// It is called by the clientside callback in the VoltageControlRow component.

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.custom = {
    blurInput: function(triggerValue, inputIdObject) {
        // triggerValue is the value from blur_trigger_id.children (e.g., n_submit)
        // inputIdObject is the dictionary ID of the dbc.Input, e.g., {"type": "...", "index": "..."}

        if (triggerValue && inputIdObject) {
            try {
                // Dash uses JSON.stringify internally for dictionary IDs to create HTML element IDs
                const element = document.getElementById(inputIdObject);
                if (element) {
                    element.blur();
                } else {
                    console.warn(
                        "Clientside blurInput: Element not found with ID string:",
                        elementIdString,
                        "derived from object:",
                        inputIdObject
                    );
                }
            } catch (e) {
                console.error("Clientside blurInput error:", e);
            }
        }
        // This clientside callback has a dummy Output.
        // We can return the trigger value or a timestamp.
        return triggerValue;
    }
};
// You can add this line for debugging to see if the file is loaded:
console.log("custom_clientside.js loaded and window.dash_clientside.custom defined:", window.dash_clientside.custom);