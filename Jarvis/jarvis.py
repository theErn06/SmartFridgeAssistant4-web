#!/usr/bin/env python3

import os
import json
import socket
import threading
import time
import re
from http.server import ThreadingHTTPServer

# Import all the backend logic from your newly created file
import functionsJarvis as fj

if __name__ == "__main__":
    print(f"--- Jarvis Unified Server ---")
    
    # 1. Thread the hardware watchers
    threading.Thread(target=fj.listen_door_events, daemon=True).start()
    threading.Thread(target=fj.door_reminder_watchdog, daemon=True).start()

    # 2. Extract Local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # 3. Create the server function and push it to a background thread so it NEVER blocks
    def start_server():
        # WE NOW CHANGE THE DIRECTORY TO THE 'Website' FOLDER SO THE BROWSER CAN SEE IT
        try:
            os.chdir(fj.WEB_DIR)
        except FileNotFoundError:
            print(f"\n[❌ FATAL ERROR] Could not find the 'Website' folder at: {fj.WEB_DIR}")
            print("Please ensure your 'Jarvis' folder and your 'Website' folder are sitting next to each other.")
            os._exit(1)
            
        print(f"\n   [🌐] Unified Web Server & API active!")
        print(f"   [📱] Your PC IP Address is: {local_ip}")
        server = ThreadingHTTPServer(('0.0.0.0', 5000), fj.JarvisAPIHandler)
        server.serve_forever()
        
    threading.Thread(target=start_server, daemon=True).start()

    # 4. Microphone Check
    try:
        import sounddevice as sd
        fj.CURRENT_MIC_NAME = sd.query_devices(kind='input')['name']
        print(f"   [MIC] Connected to: {fj.CURRENT_MIC_NAME}")
    except Exception:
        pass

    # 5. Boot sequence: Speak first, THEN display the IP on the Arduino
    fj.speak("System Initialized.")
    fj.send_to_lcd(f"IP: {local_ip}")
    fj.lcd_busy_until = time.time() + 30.0 # Lock screen for 30 seconds so you can read it!
    
    try:
        while True:
            text = ""
            
            if fj.VOICE_MODE:
                text = fj.listen_dynamic()
            else:
                if time.time() >= fj.lcd_busy_until: fj.send_to_lcd("Mode: TEXT")
                try: 
                    text = input("\n   👉 Type here: ")
                except EOFError: 
                    break

            if not text: 
                if fj.VOICE_MODE:
                    if time.time() >= fj.lcd_busy_until: fj.send_to_lcd("Listening..." if fj.jarvis_awake else "Awaiting...")
                continue
            
            fj.lcd_busy_until = 0 # Immediately release screen lock when interaction starts
            
            text_clean = fj.clean_stt_text(text)
            text_lower = text_clean.lower()
            fj.send_to_lcd(f"YOU: {text_clean}")
            
            if any(x in text_lower for x in ["switch to text", "text mode"]):
                if fj.VOICE_MODE:
                    fj.VOICE_MODE = False
                    print("\n   [SYSTEM] Switched to TEXT MODE.")
                    fj.speak("Switched to text mode.")
                continue
            
            if any(x in text_lower for x in ["switch to voice", "voice mode"]):
                if not fj.VOICE_MODE:
                    fj.VOICE_MODE = True
                    fj.jarvis_awake = True  
                    fj.conversation_ended_time = None
                    print("\n   [SYSTEM] Switched to VOICE MODE.")
                    fj.speak("Switched to voice mode. I am ready.")
                continue

            if fj.VOICE_MODE and not fj.jarvis_awake:
                
                # --- MASSIVE WAKE WORD VARIATION LIST ---
                wake_phrases = [
                    "hey jarvis", "jarvis", "jervis", "travis", "garbage", "jar", "hey joe",
                    "jobbies", "jobbie", "chavies", "java", "drop is", "joggies", "jiaobies", "ciao peace",
                    "hey jervis", "hey travis", "hay jarvis", "hey garbage", "pay jarvis",
                    "hey drivers", "hey marvis", "hey harvest", "hey service", "hey nervous",
                    "hey novice", "hey tervis", "hey dervis", "hey purvis", "a jarvis",
                    "hi jarvis", "high jarvis", "hey chaves", "hey jobs", "hey java",
                    "hey jovis", "hey javis", "hey joggers", "hey chalmers", "hey gervais",
                    "hey charters", "hey markers", "hey darvis", "hey garvis", "hey parvis",
                    "hey tarvis", "hey zarvis", "hey narvis", "hey farvis", "hey carvis",
                    "hey barvis", "hey larvis", "hey arvis", "hey orvis", "hey jabez"
                ]
                
                print(f"\n   [💤 Heard while asleep]: {text_clean}")
                
                if any(re.search(rf'\b{phrase}\b', text_lower) for phrase in wake_phrases):
                    fj.jarvis_awake = True
                    fj.conversation_ended_time = None
                    print("   [🗣️] Waking up! Jarvis is listening...")
                    fj.speak("I'm ready to listen.") 
                continue

            if any(x in text_lower for x in ["goodbye", "exit", "go to sleep"]):
                fj.speak("Thank you. Have a nice day.")
                fj.jarvis_awake = False
                fj.conversation_ended_time = time.time()
                continue

            if fj.VOICE_MODE: print(f"\n   ⏳ Processing...")
            fj.send_to_lcd("Processing...")
            
            system_instructions = (
                "You are a smart fridge. Reply ONLY in JSON array format. "
                "Valid actions: 'add', 'remove', 'list', 'lookup', 'summarize', 'chitchat'. "
                "If the user asks about waste, expired food, or a report, output exactly: "
                "[{\"action\": \"summarize\", \"period\": \"weekly\"}] (period: weekly, monthly, or all). "
                "Do not include plain text outside the JSON."
            )
            json_response = fj.ask_llm(text_clean, system_instructions) 
            
            try: fj.execute(json.loads(json_response), text_clean)
            except Exception as e: 
                print(f"Exec Error: {e}")
                fj.speak("Processing Error.")

    except KeyboardInterrupt:
        print("\nSystem terminated")
        os._exit(0)