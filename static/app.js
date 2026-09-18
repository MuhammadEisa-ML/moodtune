// When one audio preview starts, pause and reset all other previews.
document.addEventListener(
    "play",
    function (event) {
        if (event.target.tagName !== "AUDIO") {
            return;
        }

        const currentAudio = event.target;
        const allAudioPlayers = document.querySelectorAll("audio");

        allAudioPlayers.forEach(function (audio) {
            if (audio !== currentAudio) {
                audio.pause();
                audio.currentTime = 0;
            }
        });
    },
    true
);

document.querySelectorAll(".feedback-button").forEach(function (button) {
    button.addEventListener("click", async function () {
        const songCard = button.closest(".song-item");
        const status = songCard.querySelector(".feedback-status");
        const buttons = songCard.querySelectorAll(".feedback-button");

        try {
            const response = await fetch("/feedback", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    title: button.dataset.title,
                    artist: button.dataset.artist,
                    feedback: button.dataset.feedback
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error);
            }

            buttons.forEach(function (feedbackButton) {
                feedbackButton.disabled = true;
            });

            status.textContent = button.dataset.feedback === "like"
                ? "Saved as liked."
                : "Saved as not for me.";

        } catch (error) {
            status.textContent = "Feedback could not be saved.";
        }
    });
});