import cv2
import os
import time

def find_camera():
    """Find an available camera and determine the maximum safe resolution."""
    print("Searching for camera...")
    
    # Common resolutions to test (from highest to lowest)
    test_resolutions = [
        (4096, 3072),  # 4K
        (3264, 2448),  # 8MP
        (2592, 1944),  # 5MP
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (800, 600),    # SVGA
        (640, 480)     # VGA
    ]
    
    for i in range(10):
        print(f"Testing camera {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow for Windows
        
        if cap.isOpened():
            max_w, max_h = 640, 480  # Default if no high resolution works
            
            # Test resolutions from highest to lowest
            for test_w, test_h in test_resolutions:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_h)
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Check whether the resolution was actually applied
                        if actual_w >= test_w * 0.9 and actual_h >= test_h * 0.9:
                            max_w, max_h = actual_w, actual_h
                            print(f"✓ Resolution {actual_w}x{actual_h} works")
                            break
                except:
                    continue
            
            # Set the best resolution found
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_h)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera {i} working - Max resolution: {max_w}x{max_h}")
                
                # Show a preview for confirmation
                try:
                    preview = cv2.resize(frame, (800, 600))
                    cv2.putText(preview, f"Resolucao maxima: {max_w}x{max_h}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Preview Camera - Esta camera? (s/n)', preview)
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()
                    
                    if key == ord('s'):
                        cap.release()
                        return i, max_w, max_h
                except:
                    print("Preview error, but camera works")
                    cap.release()
                    return i, max_w, max_h
                
            cap.release()
        else:
            print(f"✗ Camera {i} cannot be opened")
    
    print("No camera found!")
    return None, 0, 0

def capture_photos_manual():
    """Capture photos manually at maximum resolution."""
    
    # Find camera
    cam_idx, max_width, max_height = find_camera()
    if cam_idx is None:
        print("Error: No camera found!")
        return
    
    print(f"Using camera {cam_idx}")
    print(f"Max resolution: {max_width}x{max_height}")
    
    # Configure camera for maximum resolution
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error opening camera!")
        return
    
    # Create folder for photos
    photos_folder = "photos"
    if not os.path.exists(photos_folder):
        os.makedirs(photos_folder)
        print(f"Folder created: {os.path.abspath(photos_folder)}")
    
    print(f"\n=== MANUAL PHOTO CAPTURE ===")
    print(f"Resolution: {max_width}x{max_height}")
    print("Controls:")
    print("- SPACE: Capture photo")
    print("- 'q': Exit")
    
    img_counter = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error capturing frame!")
            break
        
        # Show reduced preview (to save processing)
        preview = cv2.resize(frame, (800, 600))
        cv2.putText(preview, f"Resolucao: {max_width}x{max_height}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Fotos: {img_counter}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, "ESPACO=capturar, Q=sair", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Camera - Resolucao Maxima', preview)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}_{img_counter:03d}.jpg"
            filepath = os.path.join(photos_folder, filename)
            
            print(f"\nCapturing: {filename}")
            print(f"Frame size: {frame.shape}")
            
            # Save image in high quality
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✓ Photo {img_counter + 1} saved - {file_size} bytes")
                img_counter += 1
            else:
                print("✗ Failed to save")
                
        elif key == ord('q'):  # Exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show summary
    print(f"\n=== SUMMARY ===")
    print(f"Total photos captured: {img_counter}")
    print(f"Photo resolution: {max_width}x{max_height}")
    print(f"Folder: {os.path.abspath(photos_folder)}")

def reconnect_camera(cam_idx, max_width, max_height, retries=3):
    """Reconnect camera with multiple attempts."""
    for attempt in range(retries):
        print(f"Trying to reconnect camera... (attempt {attempt + 1}/{retries})")
        
        try:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Configure resolution safely
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Wait for stabilization
                time.sleep(0.5)
                
                # Test whether frame capture works
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print("✓ Camera reconnected successfully!")
                    return cap
                else:
                    cap.release()
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
        
        time.sleep(1)  # Wait before the next attempt
    
    print("✗ Failed to reconnect camera after multiple attempts")
    return None

def capture_photos_sequence():
    """Capture multiple photos in sequence automatically with robust reconnection."""
    
    # Find camera
    cam_idx, max_width, max_height = find_camera()
    if cam_idx is None:
        print("Error: No camera found!")
        return
    
    print(f"Using camera {cam_idx}")
    print(f"Max resolution: {max_width}x{max_height}")
    
    # Configure capture parameters
    print("\n=== CAPTURE SETTINGS ===")
    try:
        num_photos = int(input("Quantas fotos deseja capturar? "))
        interval = float(input("Intervalo entre fotos (segundos)? "))
    except ValueError:
        print("Invalid values! Using defaults: 5 photos with a 2-second interval")
        num_photos = 5
        interval = 2.0
    
    # Configure camera for maximum resolution
    cap = reconnect_camera(cam_idx, max_width, max_height)
    if cap is None:
        print("Error: Could not connect to camera!")
        return
    
    # Create folder for photos
    photos_folder = "photos_sequence"
    if not os.path.exists(photos_folder):
        os.makedirs(photos_folder)
        print(f"Folder created: {os.path.abspath(photos_folder)}")
    
    print(f"\nConfiguration:")
    print(f"- Number of photos: {num_photos}")
    print(f"- Interval: {interval} seconds")
    print(f"- Estimated total time: {num_photos * interval:.1f} seconds")
    print(f"- Resolution: {max_width}x{max_height}")
    
    # Confirmation
    confirm = input("\nDeseja continuar? (s/n): ").lower()
    if confirm != 's':
        print("Capture canceled.")
        cap.release()
        return
    
    print("\n=== AUTOMATIC CAPTURE ===")
    print("Press ESC at any time to cancel")
    
    # Initial countdown
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if ret:
            preview = cv2.resize(frame, (800, 600))
            cv2.putText(preview, f"Iniciando em: {i}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Captura Sequencial - Preparando...', preview)
        
        key = cv2.waitKey(1000) & 0xFF
        if key == 27:  # ESC
            print("Capture canceled by user")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Capture photos
    captured_count = 0
    failed_count = 0
    consecutive_failures = 0
    
    for photo_num in range(num_photos):
        print(f"\n--- Photo {photo_num + 1}/{num_photos} ---")
        
        # Check camera health before countdown
        if consecutive_failures >= 3:
            print("Too many consecutive failures. Trying to reconnect camera...")
            cap.release()
            cv2.destroyAllWindows()
            
            cap = reconnect_camera(cam_idx, max_width, max_height)
            if cap is None:
                print("✗ Reconnection failed. Stopping capture.")
                break
            consecutive_failures = 0
        
        # Countdown for this photo
        countdown_success = True
        start_time = time.time()
        while time.time() - start_time < interval:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Preview error - trying to continue...")
                countdown_success = False
                break
            
            elapsed = time.time() - start_time
            countdown = int(interval - elapsed)
            
            # Show preview with countdown
            try:
                preview = cv2.resize(frame, (800, 600))
                cv2.putText(preview, f"Foto {photo_num + 1}/{num_photos}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(preview, f"Captura em: {countdown}s", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(preview, "ESC = Cancelar", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Captura Sequencial - Preparando...', preview)
            except:
                print("Preview error - continuing...")
            
            key = cv2.waitKey(100) & 0xFF  # Reduced to 100ms for better responsiveness
            if key == 27:  # ESC
                print("Capture canceled by user")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # If there was an error during countdown, try reconnecting
        if not countdown_success:
            print("Error during countdown. Trying to reconnect...")
            cap.release()
            cap = reconnect_camera(cam_idx, max_width, max_height)
            if cap is None:
                print("✗ Reconnection failed. Skipping this photo.")
                failed_count += 1
                consecutive_failures += 1
                continue
        
        # Capture the photo with multiple attempts
        capture_success = False
        for attempt in range(3):  # 3 attempts for each photo
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                capture_success = True
                break
            else:
                print(f"Attempt {attempt + 1}/3 failed, trying again...")
                time.sleep(0.5)
        
        if not capture_success:
            print("Error capturing frame after multiple attempts!")
            failed_count += 1
            consecutive_failures += 1
            continue
        
        consecutive_failures = 0  # Reset failure counter
        
        # Generate file name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"seq_{timestamp}_{photo_num:03d}.jpg"
        filepath = os.path.join(photos_folder, filename)
        
        print(f"Capturing: {filename}")
        print(f"Frame size: {frame.shape}")
        
        # Save image in high quality
        success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success and os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✓ Photo saved: {file_size} bytes")
            captured_count += 1
            
            # Show visual feedback
            try:
                preview = cv2.resize(frame, (800, 600))
                cv2.putText(preview, "FOTO CAPTURADA!", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow('Captura Sequencial - Preparando...', preview)
                cv2.waitKey(500)  # Show for 0.5 seconds
            except:
                print("Visual feedback error - continuing...")
            
        else:
            print("✗ Failed to save photo!")
            failed_count += 1
            consecutive_failures += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print(f"\n=== FINAL REPORT ===")
    print(f"Requested photos: {num_photos}")
    print(f"Captured photos: {captured_count}")
    print(f"Failed photos: {failed_count}")
    print(f"Success rate: {(captured_count/num_photos)*100:.1f}%")
    print(f"Photo resolution: {max_width}x{max_height}")
    print(f"Folder: {os.path.abspath(photos_folder)}")
    
    if captured_count > 0:
        print("Captured photos:")
        import glob
        saved_files = glob.glob(os.path.join(photos_folder, "seq_*"))
        for f in sorted(saved_files):
            size = os.path.getsize(f)
            print(f"  - {os.path.basename(f)} ({size} bytes)")

def capture_photos_sequence_fast():
    """Capture photos in sequence WITHOUT preview (faster and more stable)."""
    
    # Find camera
    cam_idx, max_width, max_height = find_camera()
    if cam_idx is None:
        print("Error: No camera found!")
        return
    
    print(f"Using camera {cam_idx}")
    print(f"Max resolution: {max_width}x{max_height}")
    
    # Configure capture parameters
    print("\n=== FAST CAPTURE SETTINGS ===")
    try:
        num_photos = int(input("Quantas fotos deseja capturar? "))
        interval = float(input("Intervalo entre fotos (segundos)? "))
    except ValueError:
        print("Invalid values! Using defaults: 10 photos with a 2-second interval")
        num_photos = 10
        interval = 2.0
    
    # Create folder for photos
    photos_folder = "photos_fast_sequence"
    if not os.path.exists(photos_folder):
        os.makedirs(photos_folder)
        print(f"Folder created: {os.path.abspath(photos_folder)}")
    
    print(f"\nConfiguration:")
    print(f"- Number of photos: {num_photos}")
    print(f"- Interval: {interval} seconds")
    print(f"- Resolution: {max_width}x{max_height}")
    print(f"- Mode: WITHOUT preview (more stable)")
    
    # Confirmation
    confirm = input("\nDeseja continuar? (s/n): ").lower()
    if confirm != 's':
        print("Capture canceled.")
        return
    
    print("\n=== FAST CAPTURE ===")
    print("IMPORTANT: No preview! Position the camera now!")
    input("Pressione ENTER quando estiver pronto...")
    
    # Capture photos
    captured_count = 0
    failed_count = 0
    
    for photo_num in range(num_photos):
        print(f"\nPhoto {photo_num + 1}/{num_photos}...")
        
        # Connect camera only for this photo
        try:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"✗ Error configuring camera: {e}")
            failed_count += 1
            continue
        
        if not cap.isOpened():
            print("✗ Error opening camera")
            failed_count += 1
            continue
        
        # Wait for camera stabilization
        time.sleep(0.5)
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()  # Release immediately
        
        if not ret or frame is None:
            print("✗ Error capturing frame")
            failed_count += 1
            continue
        
        # Save photo
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"fast_{timestamp}_{photo_num:03d}.jpg"
        filepath = os.path.join(photos_folder, filename)
        
        success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success and os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✓ Saved: {file_size} bytes")
            captured_count += 1
        else:
            print("✗ Failed to save")
            failed_count += 1
        
        # Wait interval (except after the last photo)
        if photo_num < num_photos - 1:
            print(f"Waiting {interval}s...")
            time.sleep(interval)
    
    # Final report
    print(f"\n=== FINAL REPORT ===")
    print(f"Requested photos: {num_photos}")
    print(f"Captured photos: {captured_count}")
    print(f"Failed photos: {failed_count}")
    print(f"Success rate: {(captured_count/num_photos)*100:.1f}%")
    print(f"Folder: {os.path.abspath(photos_folder)}")

if __name__ == "__main__":
    print("=== PHOTO CAPTURE ===")
    print("1. Capture photos manually (SPACE to capture)")
    print("2. Capture multiple photos in sequence (with preview)")
    print("3. Capture multiple photos in sequence (fast/stable mode)")
    
    choice = input("Escolha uma opção (1/2/3): ").strip()
    
    if choice == "1":
        capture_photos_manual()
    elif choice == "2":
        capture_photos_sequence()
    elif choice == "3":
        capture_photos_sequence_fast()
    else:
        print("Invalid option! Using manual capture...")
        capture_photos_manual()