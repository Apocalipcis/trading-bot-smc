#!/usr/bin/env python3
"""
Test hotkeys functionality
"""
import time
try:
    import keyboard
    print("✅ Keyboard library imported successfully")
    
    print("🧪 Testing hotkey detection...")
    print("Press 'T' to test signal, 'Q' to quit")
    
    start_time = time.time()
    while time.time() - start_time < 10:  # Test for 10 seconds
        try:
            if keyboard.is_pressed('t'):
                print("🔥 T key detected!")
                time.sleep(0.5)  # Debounce
            elif keyboard.is_pressed('q'):
                print("👋 Q key detected, exiting...")
                break
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error in keyboard detection: {e}")
            break
    
    print("⏰ Test completed")
    
except ImportError:
    print("❌ Keyboard library not available")
except Exception as e:
    print(f"❌ Error: {e}")
