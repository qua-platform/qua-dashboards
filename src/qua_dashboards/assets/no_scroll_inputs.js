// Prevents the mouse wheel from changing the value of number inputs marked with
// the "no-scroll-input" class. Browsers only apply wheel steps to a focused number
// input, so blurring it on wheel keeps the value fixed and lets the page scroll.

document.addEventListener(
    "wheel",
    function (event) {
        const target = event.target;
        if (
            target instanceof HTMLInputElement &&
            target.classList.contains("no-scroll-input") &&
            document.activeElement === target
        ) {
            target.blur();
        }
    },
    { capture: true, passive: true }
);
