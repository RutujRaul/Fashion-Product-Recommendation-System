document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    form.addEventListener("submit", function() {
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    });
});
