import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

class CrexSpeaker:
    def __init__(self, voice="slt", player="aplay"):
        """
        Initialize the speaker.
        Available voices: kal, awb, rms, slt, etc.
        Available players: aplay, paplay, ffplay -nodisp -autoexit
        """
        self.voice = voice
        self.player = player

    def speak(self, text: str):
        """Synthesize and play the text."""
        if not text:
            return

        # Create a temporary file for the speech
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_wave = tf.name

        try:
            # 1. Synthesize text to WAV
            # flite -voice SLT -t "text" -o output.wav
            logger.info(f"Synthesizing: {text[:50]}...")
            synth_cmd = ["flite", "-voice", self.voice, "-t", text, "-o", temp_wave]
            subprocess.run(synth_cmd, check=True, capture_output=True)

            # 2. Play the WAV file
            # aplay output.wav
            play_cmd = [self.player, temp_wave]
            subprocess.run(play_cmd, check=True, capture_output=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"Speaker error: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            logger.error(f"Unexpected speaker error: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_wave):
                os.remove(temp_wave)

# Singleton instance
speaker = CrexSpeaker()

if __name__ == "__main__":
    # Test the speaker
    logging.basicConfig(level=logging.INFO)
    s = CrexSpeaker(voice="slt")
    s.speak("Hello Sir. My voice is now active. I am ready to assist with your mission.")
