#!/usr/bin/env python3
"""
Test script to verify the connection between Flask backend and React frontend
"""

import requests
import socketio
import time

def test_flask_backend():
    """Test if Flask backend is running"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        print("Flask backend is running on http://localhost:5000")
        print(f"   Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Flask backend is not running: {e}")
        return False

def test_react_frontend():
    """Test if React frontend is running"""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        print("React frontend is running on http://localhost:3000")
        print(f"   Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"React frontend is not running: {e}")
        return False

def test_socketio_connection():
    """Test Socket.IO connection"""
    try:
        sio = socketio.Client()
        
        @sio.event
        def connect():
            print("Socket.IO connection established")
            sio.disconnect()
            
        @sio.event
        def disconnect():
            print("Socket.IO connection closed")
            
        sio.connect('http://localhost:5000')
        time.sleep(1)
        return True
    except Exception as e:
        print(f"Socket.IO connection failed: {e}")
        return False

def main():
    print("Testing Sign2Speak Connection...")
    print("=" * 50)
    
    # Test Flask backend
    flask_ok = test_flask_backend()
    print()
    
    # Test React frontend
    react_ok = test_react_frontend()
    print()
    
    # Test Socket.IO connection
    socketio_ok = test_socketio_connection()
    print()
    
    print("=" * 50)
    if flask_ok and react_ok and socketio_ok:
        print("All connections successful!")
        print("You can now:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Click 'Start Recognition' in the left panel")
        print("   3. Show hand signs to your camera")
        print("   4. See real-time sign language recognition!")
    else:
        print("Some connections failed. Please check:")
        if not flask_ok:
            print("   - Start Flask backend: cd sign2text && python app.py")
        if not react_ok:
            print("   - Start React frontend: cd frontend && npm run dev")
        if not socketio_ok:
            print("   - Check Socket.IO configuration")

if __name__ == "__main__":
    main()
