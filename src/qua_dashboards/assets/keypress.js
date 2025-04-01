console.log("Testing keypress.js");

// document.addEventListener("DOMContentLoaded", function() {
//     // Wait for the DOM to fully load
//     document.addEventListener("keydown", function(event) {
//         if (event.shiftKey) {
//             let key = event.key.toLowerCase();  // Normalize to lowercase

//             let mapping = {
//                 "p": "point",
//                 "l": "line",
//                 "d": "delete"
//             };

//             if (mapping.hasOwnProperty(key)) {
//                 event.preventDefault();  // Prevent default browser actions (e.g., print dialog)

//                 let selectedValue = mapping[key];

//                 // Make sure the store element exists before updating it
//                 let store = document.getElementById("key-store");

//                 // Log the element to check if it exists
//                 console.log(store);  // Check if it is null or undefined

//                 if (store) {
//                     store.data = selectedValue;  // Update the data attribute of the store
//                 } else {
//                     console.error("key-store element not found");
//                 }

//                 // Optionally, dispatch the custom event as you were doing before
//                 // let radioEvent = new CustomEvent("radio-keypress", { detail: selectedValue });
//                 // document.dispatchEvent(radioEvent);
//             }
//         }
//     });
// });

document.addEventListener("keydown", function(event) {
    // Ensure that Shift is held when another key is pressed
    if (event.shiftKey) {
        let key = event.key.toLowerCase();  // Normalize to lowercase
        console.log("shiftKey:", event.shiftKey)
        console.log("key:", key)

        // Define a mapping for specific keys when Shift is held
        let mapping = {
            "p": "point",
            "l": "line",
            "d": "delete"
        };

        // Check if the pressed key exists in the mapping and handle it
        if (mapping.hasOwnProperty(key)) {
            event.preventDefault();  // Prevent default browser actions (e.g., print dialog)

            // Get the corresponding action and dispatch the custom event
            let selectedValue = mapping[key];
            console.log("selectedValue:", selectedValue)

            // Update the dcc.Store component with the selected value
            let store = document.getElementById('key-store');  // Reference to dcc.Store by id
            store.data = selectedValue;  // Update the data attribute of the store

            // let radioEvent = new CustomEvent("radio-keypress", { detail: selectedValue });
            // document.dispatchEvent(radioEvent);
        }
    }
});


// document.addEventListener("keydown", function(event) {
//     console.log("Raw event key:", event.key);  // Check actual key output
// });

// document.addEventListener("keydown", function(event) {
//     // Check if Shift is pressed along with another key (e.g., P, L, D)
//     if (event.shiftKey) {
//         let keyCombo = event.key.toLowerCase();  // Convert key to lowercase for uniformity

//         console.log("Detected key with Shift:", keyCombo);

//         if (keyCombo === "p" || keyCombo === "l" || keyCombo === "d") {
//             event.preventDefault();  // Prevent default action (like print or other actions)
//             console.log("Prevented default action for:", "Shift+" + keyCombo.toUpperCase());
//         }
//     }
// });

// document.addEventListener("keydown", function(event) {
//     if (event.shiftKey) {  // Check if SHIFT is held
//         let key = event.key.toLowerCase();  // Normalize to lowercase

//         let mapping = {
//             "p": "point",
//             "l": "line",
//             "d": "delete"
//         };

//         if (key in mapping) {
//             event.preventDefault();  // Prevent browser default actions (e.g., print dialog)
//             let selectedValue = mapping[key];

//             // Dispatch custom event with selected value
//             let radioEvent = new CustomEvent("radio-keypress", { detail: selectedValue });
//             document.dispatchEvent(radioEvent);
//         }
//     }
// });



// document.addEventListener("keydown", function (event) {
//     let keyCombo = "";
    
//     // Check if CTRL is pressed
//     if (event.shiftKey) keyCombo += "Shift+"; // if CTRL is pressed, append Control+ to keyCombo
    
//     // Capture the main key
//     keyCombo += event.key.toUpperCase();

//     console.log(keyCombo);  // Add this line after assigning keyCombo

//     // // Prevent the default action (i.e., prevent printing for CTRL + P)
//     // if (keyCombo === "Control+P" || keyCombo === "Control+L" || keyCombo === "Control+D") {
//     //     event.preventDefault();  // Prevents opening the print dialog
//     // }
    
//     // Send the key press combination to Dash (updating dcc.Store)
//     let keyStore = document.getElementById("key-store");
    
//     // Only proceed if the dcc.Store component exists in the layout
//     if (keyStore) {
//         let eventData = { "data": keyCombo };  // The key combination to pass
//         keyStore.setAttribute("data-dash-data", JSON.stringify(eventData));  // pass the keyCombo in JSON format
//     }
// });