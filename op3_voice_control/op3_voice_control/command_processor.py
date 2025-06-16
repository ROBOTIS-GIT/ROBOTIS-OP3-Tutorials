#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Twist
import re
import threading
import time
import subprocess
import random
from gtts import gTTS
import pygame
import io

class CommandProcessor(Node):
    def __init__(self):
        super().__init__('command_processor')
        
        # TTS engine settings (using gTTS for more natural voice)
        self.tts_method = "gtts"  # "festival", "gtts", or "espeak"
        
        # pygame initialization (for audio playback)
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        # TTS state management (enhanced echo prevention)
        self.is_speaking = False
        self.tts_lock = threading.Lock()
        self.last_tts_time = 0  # Track last TTS time
        self.tts_cooldown = 2.0  # 2 second wait after TTS completion
        self.recent_tts_texts = []  # Recent TTS text history
        
        # Command execution state management (prevent duplicate commands)
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 1.0  # 1 second cooldown between same commands
        
        # Subscribers
        self.voice_subscriber = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10)
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_publisher = self.create_publisher(Int32, '/robotis/action/page_num', 10)
        self.mic_control_publisher = self.create_publisher(String, '/mic_control', 10)
        self.mode_command_publisher = self.create_publisher(String, '/robotis/mode_command', 10)
        
        # Command pattern definitions
        self.command_patterns = {
            'ready_mode': [r'ready', r'ready mode', r'standby', r'standby mode'],
            'motion_mode': [r'action', r'motion', r'action mode', r'motion mode'],
            'stop': [r'stop', r'halt', r'freeze'],
            'stand_up': [r'stand up', r'get up', r'standup'],
            'sit_down': [r'sit down', r'sit', r'sitdown'],
            'bye_bye': [r'bye', r'goodbye', r'see you later'],
            'thank_you': [r'thank you', r'thanks', r'appreciate it'],
            'oops': [r'oops', r'whoops', r'sorry', r'my bad'],
            'wow': [r'wow', r'amazing', r'impressive'],
            'clap': [r'clap', r'applaud', r'cheer'],
            'status': [r'status', r'how are you', r'state']
        }
        
        self.get_logger().info("Command processor started with gTTS (high quality)")
    
    def speak_festival(self, text):
        """Festival TTS for speech synthesis (high quality voice)"""
        try:
            # Execute Festival command with high quality voice
            festival_script = f'(voice_kal_diphone)\n(SayText "{text}")'
            process = subprocess.Popen(['festival'], 
                                     stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
            process.communicate(input=festival_script)
        except Exception as e:
            self.get_logger().error(f"Festival TTS error: {e}")
    
    def speak_espeak(self, text):
        """espeak TTS for speech synthesis (fast and stable)"""
        try:
            # Execute espeak command for TTS
            subprocess.run(['espeak', '-s', '150', '-v', 'en+f3', text], 
                         check=False)
        except Exception as e:
            self.get_logger().error(f"espeak TTS error: {e}")
    
    def speak_gtts(self, text):
        """Google TTS for speech synthesis (most natural voice)"""
        try:
            # Generate speech with gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to memory buffer
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Play with pygame
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            
            # Wait until playback is complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            self.get_logger().error(f"gTTS error: {e}")
            # Fallback to Festival if gTTS fails
            self.speak_festival(text)
    
    def voice_command_callback(self, msg):
        """Voice command processing"""
        # Ignore voice commands while TTS is active
        if self.is_speaking:
            self.get_logger().debug("Ignoring voice command while speaking")
            return
            
        # Check TTS cooldown period
        current_time = time.time()
        if current_time - self.last_tts_time < self.tts_cooldown:
            self.get_logger().debug(f"Ignoring command during TTS cooldown: {current_time - self.last_tts_time:.1f}s")
            return
            
        command_text = msg.data.lower().strip()
        
        if not command_text:
            return
        
        # Check if it's TTS echo (enhanced filtering)
        if self.is_tts_echo(command_text):
            self.get_logger().debug(f"Filtered TTS echo: {command_text}")
            return
        
        # Check similarity with recent TTS texts
        if self.is_similar_to_recent_tts(command_text):
            self.get_logger().debug(f"Filtered similar to recent TTS: {command_text}")
            return
        
        self.get_logger().info(f"Processing command: {command_text}")
        
        # Command matching
        matched_command = self.match_command(command_text)
        
        if matched_command:
            self.get_logger().info(f"Matched command: '{command_text}' -> '{matched_command}'")
            self.execute_command(matched_command, command_text)
        else:
            self.get_logger().warning(f"No command matched for: {command_text}")
            self.handle_unknown_command(command_text)
    
    def is_tts_echo(self, text):
        """Check if it's TTS echo"""
        # Filter common TTS response patterns - use more specific patterns
        tts_patterns = [
            r'^stopping$',  # Only exact match for "stopping"
            r'^standing up$',  # Only exact match to avoid filtering voice commands
            r'^sitting down$', # Only exact match to avoid filtering voice commands
            r'entering ready mode',
            r'entering motion mode',
            r'goodbye.*see you later',
            r'you\'re welcome',
            r'no problem.*happens to everyone',
            r'thank you.*glad.*impressed',  # Keep this pattern for TTS responses
            r'thank you.*applause',         # Keep this pattern for TTS responses
            r'operational and ready',
            r'didn\'t understand',
            r'please repeat',
            r'not sure what',
            r'try saying',
            r'couldn\'t execute'
        ]
        
        for pattern in tts_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def is_similar_to_recent_tts(self, text):
        """Check if similar to recent TTS texts"""
        # Compare with recent TTS texts
        for recent_text in self.recent_tts_texts:
            # Word-level similarity check
            if self.text_similarity(text, recent_text) > 0.7:
                return True
        return False
    
    def text_similarity(self, text1, text2):
        """Calculate similarity between two texts (0.0-1.0)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def match_command(self, text):
        """Command pattern matching"""
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    self.get_logger().info(f"Pattern match found: '{text}' -> '{command}'")
                    return command
        
        return None
    
    def execute_command(self, command, original_text):
        """Execute command with duplicate prevention"""
        # Check for duplicate commands within cooldown period
        current_time = time.time()
        
        # Different cooldown policies for different commands
        if command == 'stop':
            # Stop commands can be more frequent (shorter cooldown)
            cooldown_period = 0.5
        else:
            # Other commands use longer cooldown
            cooldown_period = self.command_cooldown
            
        if (self.last_command == command and 
            current_time - self.last_command_time < cooldown_period):
            self.get_logger().debug(f"Ignoring duplicate command '{command}' within {cooldown_period}s cooldown period")
            return
        
        # Update command tracking
        self.last_command = command
        self.last_command_time = current_time
        
        response = ""
        
        try:
            if command == 'ready_mode':
                self.send_mode_command("ready")
                response = "Entering ready mode"
                
            elif command == 'motion_mode':
                self.send_mode_command("action")
                response = "Entering motion mode"

            elif command == 'stop':
                self.send_action_page(-1)  # Stop action
                response = "Stopping"
                
            elif command == 'stand_up':
                self.get_logger().info("Executing STAND UP command")
                self.send_action_page(1)  # Stand up action
                response = "Standing up"
                
            elif command == 'sit_down':
                self.send_action_page(15)  # Sit down action
                response = "Sitting down"
                
            elif command == 'bye_bye':
                self.send_action_page(38)  # Bye bye action
                response = "Goodbye! See you later!"
                
            elif command == 'thank_you':
                self.send_action_page(4)  # Thank you action
                response = "Thank you!"
                
            elif command == 'oops':
                self.send_action_page(27)  # Oops action
                response = "Oops!"
                
            elif command == 'wow':
                self.send_action_page(24)  # Wow action
                response = "Wow!"
                
            elif command == 'clap':
                self.send_action_page(54)  # Clap action
                response = "Clap please!"
                
            elif command == 'status':
                response = "I am operational and ready for commands"
                
            else:
                response = f"Command {command} not implemented yet"
            
            # TTS response
            self.speak(response)
            
        except Exception as e:
            self.get_logger().error(f"Command execution error: {e}")
            self.speak("Sorry, I couldn't execute that command")
    
    def handle_unknown_command(self, text):
        """Handle unknown command"""
        self.get_logger().warn(f"Unknown command: {text}")
        
        responses = [
            "I didn't understand that command",
            "Could you please repeat that?",
            "I'm not sure what you mean",
            "Try saying: motion mode, stand up, sit down, or status"
        ]
        
        import random
        response = random.choice(responses)
        self.speak(response)
    
    def send_action_page(self, page_num):
        """Send action page number to /robotis/action/page_num"""
        msg = Int32()
        msg.data = int(page_num)  # Ensure it's an integer
        self.action_publisher.publish(msg)
        self.get_logger().info(f"Sent action page: {page_num} (command type: {type(page_num)})")
    
    def send_mode_command(self, mode):
        """Send mode command to demo_node"""
        msg = String()
        msg.data = mode
        self.mode_command_publisher.publish(msg)
        self.get_logger().info(f"Sent mode command: {mode}")
    
    def speak(self, text):
        """High-quality TTS speech (with echo prevention)"""
        def tts_thread():
            with self.tts_lock:
                try:
                    # Send microphone disable signal
                    self.disable_microphone()
                    
                    # Set TTS state flag
                    self.is_speaking = True
                    
                    # Add TTS text to recent history
                    self.recent_tts_texts.append(text.lower())
                    # Keep only recent 10 texts
                    if len(self.recent_tts_texts) > 10:
                        self.recent_tts_texts.pop(0)
                    
                    # Execute TTS (Festival, gTTS or espeak)
                    if self.tts_method == "festival":
                        self.speak_festival(text)
                    elif self.tts_method == "gtts":
                        self.speak_gtts(text)
                    else:
                        self.speak_espeak(text)
                    
                    # Wait briefly after TTS completion (echo prevention)
                    time.sleep(0.5)
                    
                    # Record last TTS time
                    self.last_tts_time = time.time()
                    
                except Exception as e:
                    self.get_logger().error(f"TTS error: {e}")
                    # Fallback on failure
                    try:
                        self.speak_festival(text)
                    except:
                        self.get_logger().error("All TTS methods failed")
                finally:
                    # Clear TTS state flag
                    self.is_speaking = False
                    
                    # Send microphone reactivation signal
                    self.enable_microphone()
        
        # Execute in separate thread
        thread = threading.Thread(target=tts_thread)
        thread.daemon = True
        thread.start()
    
    def disable_microphone(self):
        """Send microphone disable signal"""
        msg = String()
        msg.data = "disable"
        self.mic_control_publisher.publish(msg)
    
    def enable_microphone(self):
        """Send microphone enable signal"""
        msg = String()
        msg.data = "enable"
        self.mic_control_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommandProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()