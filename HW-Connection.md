# Hardware Connection Diagram for NiaotoShield Cut-Off Board

The following diagram shows the connections between the Raspberry Pi (main board), Arduino Nano (in the cut-off box), GPS module, and relay.

```svg
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f8f8"/>
  
  <!-- Title -->
  <text x="400" y="40" font-family="Arial" font-size="24" text-anchor="middle" font-weight="bold">NiaotoShield Cut-Off Board Connection Diagram</text>
  
  <!-- Raspberry Pi -->
  <rect x="80" y="120" width="200" height="140" rx="10" fill="#c51a4a" stroke="#333" stroke-width="2"/>
  <text x="180" y="150" font-family="Arial" font-size="18" text-anchor="middle" fill="white">Raspberry Pi 4</text>
  <text x="180" y="180" font-family="Arial" font-size="14" text-anchor="middle" fill="white">Main Board</text>
  
  <!-- Pi GPIO Pins -->
  <circle cx="110" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="110" y="240" font-family="Arial" font-size="10" text-anchor="middle">TX</text>
  
  <circle cx="140" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="140" y="240" font-family="Arial" font-size="10" text-anchor="middle">RX</text>
  
  <circle cx="170" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="170" y="240" font-family="Arial" font-size="10" text-anchor="middle">GND</text>
  
  <circle cx="200" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="200" y="240" font-family="Arial" font-size="10" text-anchor="middle">5V</text>
  
  <!-- Arduino Nano -->
  <rect x="500" y="120" width="200" height="140" rx="6" fill="#00979c" stroke="#333" stroke-width="2"/>
  <text x="600" y="150" font-family="Arial" font-size="18" text-anchor="middle" fill="white">Arduino Nano</text>
  <text x="600" y="180" font-family="Arial" font-size="14" text-anchor="middle" fill="white">Cut-Off Controller</text>
  
  <!-- Arduino Pins -->
  <circle cx="520" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="520" y="240" font-family="Arial" font-size="10" text-anchor="middle">RX(0)</text>
  
  <circle cx="550" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="550" y="240" font-family="Arial" font-size="10" text-anchor="middle">TX(1)</text>
  
  <circle cx="580" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="580" y="240" font-family="Arial" font-size="10" text-anchor="middle">GND</text>
  
  <circle cx="610" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="610" y="240" font-family="Arial" font-size="10" text-anchor="middle">5V</text>
  
  <circle cx="640" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="640" y="240" font-family="Arial" font-size="10" text-anchor="middle">D4</text>
  
  <circle cx="670" cy="220" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="670" y="240" font-family="Arial" font-size="10" text-anchor="middle">D3</text>
  
  <circle cx="520" cy="270" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="520" y="290" font-family="Arial" font-size="10" text-anchor="middle">D8</text>
  
  <circle cx="550" cy="270" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="550" y="290" font-family="Arial" font-size="10" text-anchor="middle">D9</text>
  
  <circle cx="580" cy="270" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="580" y="290" font-family="Arial" font-size="10" text-anchor="middle">D2</text>
  
  <!-- GPS Module -->
  <rect x="500" y="350" width="180" height="100" rx="5" fill="#4caf50" stroke="#333" stroke-width="2"/>
  <text x="590" y="380" font-family="Arial" font-size="16" text-anchor="middle" fill="white">GPS Module</text>
  <text x="590" y="400" font-family="Arial" font-size="12" text-anchor="middle" fill="white">TinyGPS++</text>
  
  <!-- GPS Pins -->
  <circle cx="520" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="520" y="450" font-family="Arial" font-size="10" text-anchor="middle">VCC</text>
  
  <circle cx="550" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="550" y="450" font-family="Arial" font-size="10" text-anchor="middle">GND</text>
  
  <circle cx="580" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="580" y="450" font-family="Arial" font-size="10" text-anchor="middle">TX</text>
  
  <circle cx="610" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="610" y="450" font-family="Arial" font-size="10" text-anchor="middle">RX</text>
  
  <!-- Relay -->
  <rect x="220" y="350" width="160" height="100" rx="5" fill="#3498db" stroke="#333" stroke-width="2"/>
  <text x="300" y="380" font-family="Arial" font-size="16" text-anchor="middle" fill="white">Relay Module</text>
  <text x="300" y="400" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Cut-Off Control</text>
  
  <!-- Relay Pins -->
  <circle cx="250" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="250" y="450" font-family="Arial" font-size="10" text-anchor="middle">VCC</text>
  
  <circle cx="280" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="280" y="450" font-family="Arial" font-size="10" text-anchor="middle">GND</text>
  
  <circle cx="310" cy="430" r="5" fill="#ffcc00" stroke="#333"/>
  <text x="310" y="450" font-family="Arial" font-size="10" text-anchor="middle">IN</text>
  
  <!-- LED Indicators -->
  <circle cx="400" y="350" r="15" fill="#ff9800" stroke="#333" stroke-width="2"/>
  <text x="400" y="320" font-family="Arial" font-size="12" text-anchor="middle">GPS LED (Orange)</text>
  <text x="400" y="335" font-family="Arial" font-size="12" text-anchor="middle">D2</text>
  
  <circle cx="450" y="350" r="15" fill="#2196f3" stroke="#333" stroke-width="2"/>
  <text x="450" y="320" font-family="Arial" font-size="12" text-anchor="middle">Relay LED (Blue)</text>
  <text x="450" y="335" font-family="Arial" font-size="12" text-anchor="middle">D3</text>
  
  <!-- Connection Lines -->
  <!-- Pi to Arduino UART -->
  <line x1="110" y1="220" x2="520" y2="220" stroke="#ff5722" stroke-width="2"/>
  <text x="320" y="210" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">UART TX → RX</text>
  
  <line x1="140" y1="220" x2="550" y2="220" stroke="#4caf50" stroke-width="2"/>
  <text x="320" y="235" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">UART RX ← TX</text>
  
  <!-- Ground and Power -->
  <line x1="170" y1="220" x2="580" y2="220" stroke="#333" stroke-width="2"/>
  <text x="320" y="260" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">GND</text>
  
  <line x1="200" y1="220" x2="610" y2="220" stroke="#e91e63" stroke-width="2"/>
  <text x="320" y="285" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">5V Power</text>
  
  <!-- Arduino to GPS -->
  <line x1="520" y1="270" x2="580" y2="430" stroke="#ff9800" stroke-width="2"/>
  <text x="540" y="320" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">D8 → GPS TX</text>
  
  <line x1="550" y1="270" x2="610" y2="430" stroke="#9c27b0" stroke-width="2"/>
  <text x="590" y="320" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">D9 → GPS RX</text>
  
  <line x1="610" y1="220" x2="520" y2="430" stroke="#e91e63" stroke-width="2"/>
  <text x="480" y="320" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">5V → GPS VCC</text>
  
  <line x1="580" y1="220" x2="550" y2="430" stroke="#333" stroke-width="2"/>
  <text x="500" y="340" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">GND → GPS GND</text>
  
  <!-- Arduino to Relay -->
  <line x1="640" y1="220" x2="310" y2="430" stroke="#3f51b5" stroke-width="2"/>
  <text x="450" y="340" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">D4 → Relay IN</text>
  
  <line x1="610" y1="220" x2="250" y2="430" stroke="#e91e63" stroke-width="2"/>
  <text x="400" y="320" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">5V → Relay VCC</text>
  
  <line x1="580" y1="220" x2="280" y2="430" stroke="#333" stroke-width="2"/>
  <text x="410" y="360" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">GND → Relay GND</text>
  
  <!-- Arduino to LEDs -->
  <line x1="580" y1="270" x2="400" y2="335" stroke="#ff9800" stroke-width="2"/>
  <line x1="670" y1="220" x2="450" y2="335" stroke="#2196f3" stroke-width="2"/>
  
  <!-- Security Protocol Note -->
  <rect x="100" y="500" width="600" height="70" rx="10" fill="#f5f5f5" stroke="#333" stroke-width="1"/>
  <text x="400" y="525" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold">Secure Communication Protocol</text>
  <text x="400" y="550" font-family="Arial" font-size="14" text-anchor="middle">Lightweight encrypted communication with GPS data transmission</text>
</svg>
```

## Connection Details

### Raspberry Pi to Arduino Nano
- **Pi TX (GPIO14) → Arduino RX (D0)**: Data transmission from Pi to Arduino
- **Pi RX (GPIO15) → Arduino TX (D1)**: Data transmission from Arduino to Pi
- **Pi GND → Arduino GND**: Common ground
- **Pi 5V → Arduino 5V**: Power supply for Arduino

### Arduino Nano to GPS Module
- **Arduino D8 (RX) → GPS TX**: GPS data received by Arduino
- **Arduino D9 (TX) → GPS RX**: Commands sent to GPS module (if needed)
- **Arduino 5V → GPS VCC**: Power supply for GPS module
- **Arduino GND → GPS GND**: Common ground

### Arduino Nano to Relay Module
- **Arduino D4 → Relay IN**: Control signal for relay activation
- **Arduino 5V → Relay VCC**: Power supply for relay module
- **Arduino GND → Relay GND**: Common ground

### Arduino LED Indicators
- **D2 (Pin 2)**: Orange LED for GPS status indication
- **D3 (Pin 3)**: Blue LED for relay status indication

## Important Notes

1. **Secure Communication**: All communication between Raspberry Pi and Arduino uses lightweight encryption for security.
2. **GPS Module Placement**: The GPS module requires clear view of the sky for optimal performance.
3. **Relay Connection**: The relay should be connected to the car's electrical system for cut-off functionality.
4. **LED Indicators**: The LEDs help in debugging and status monitoring (orange for GPS, blue for relay).
5. **Security**: The cut-off box should be installed in a hidden location to prevent tampering.
6. **Power Considerations**: All components should be properly powered for reliable operation.

## Testing and Setup

1. When the system powers up, the Arduino will run through a test sequence:
   - Relay activation test (blinking 5 times)
   - GPS connectivity test
   - Communication test with the Raspberry Pi

2. The Raspberry Pi runs a service that:
   - Securely communicates with the Arduino
   - Receives GPS coordinates
   - Updates the Supabase database every minute
   - Can send commands to control the relay

3. LED indicators provide visual feedback:
   - Orange LED blinks when GPS data is received
   - Blue LED turns on when relay is activated

For installation and setup instructions, refer to the `setup_gps_service.sh` script which configures the Raspberry Pi and installs the GPS service. 