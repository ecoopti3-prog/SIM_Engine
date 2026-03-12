# Arduino Thermal Calibration — Wiring Guide

## Required Hardware

| Component | Purpose | Cost | Where to buy |
|-----------|---------|------|-------------|
| Arduino Uno | Stress test platform | ~$25 | Arduino.cc, AliExpress |
| DS18B20 **or** LM35 | Temperature sensor | $1–3 | Amazon, DigiKey |
| 4.7kΩ resistor | DS18B20 pullup (if using DS18B20) | $0.05 | Any electronics store |
| USB-A to USB-B cable | Already have one | — | — |
| Breadboard + jumpers | Wiring | $3 | Amazon |

**Sensor recommendation:** DS18B20 (±0.0625°C, digital, noise-immune).  
LM35 also works (±0.1°C with Arduino ADC) and needs no library.

---

## Option A: DS18B20 Digital Sensor (recommended)

```
Arduino Uno                DS18B20
──────────────────────────────────────
Pin 2  ←───────────────── DQ  (data)
5V     ←───────────────── VDD (power)
GND    ←───────────────── GND

                  4.7kΩ pullup resistor:
                  DQ pin ──[4.7kΩ]── 5V
```

**Physical placement:**  
Lay the DS18B20 flat on top of the ATmega328P chip (the big black IC).  
If possible, use a small piece of thermal tape to hold it in contact.  
**Do NOT place it on the USB controller chip** (smaller IC near USB port).

**Library required:**  
Arduino IDE → Sketch → Include Library → Manage Libraries  
Search: `DallasTemperature` → Install (also installs OneWire dependency)

---

## Option B: LM35 Analog Sensor (no library needed)

```
Arduino Uno                LM35
──────────────────────────────────────
A0     ←───────────────── VOUT (center pin)
5V     ←───────────────── VS   (left pin, flat side facing you)
GND    ←───────────────── GND  (right pin)
```

**Placement:** Same as DS18B20 — directly on ATmega328P IC surface.

---

## Sketch Configuration

In `arduino_stress_test.ino`, line 1:

```cpp
// For DS18B20:
#define SENSOR_TYPE  2

// For LM35:
#define SENSOR_TYPE  1

// For NTC thermistor 10kΩ:
#define SENSOR_TYPE  3
```

---

## Upload & Run

1. Open `arduino_stress_test.ino` in Arduino IDE
2. Select: Tools → Board → Arduino Uno
3. Select: Tools → Port → (your port, e.g., COM3 or /dev/ttyACM0)
4. Click Upload
5. Open Serial Monitor: Tools → Serial Monitor → set baud to 115200
6. Watch the run: IDLE 2min → ACTIVE 8min
7. Save serial output to file (copy-paste or use serial logging tool)

**LED indicator:**  
- OFF = IDLE phase (CPU resting)  
- ON  = ACTIVE phase (100% CPU stress)

---

## Python Calibration

After run completes, save serial output to a file, then:

```bash
# From sim_engine directory:

# Demo mode (no hardware, synthetic data):
python calibration/fit_thermal.py --demo

# From saved file:
python calibration/fit_thermal.py --file calibration/data/my_run.txt

# Live (Arduino connected):
python calibration/fit_thermal.py --port /dev/ttyACM0
# Windows: --port COM3
```

---

## What to Expect

| Measurement | Expected value | Acceptable range |
|-------------|---------------|-----------------|
| T_idle | Room temperature (e.g., 23°C) | 15–35°C |
| T_active | T_idle + **3–5°C** | T_idle + 2–8°C |
| Time constant τ | ~50–60 seconds | 30–90s |
| R_theta measured | ~60–80°C/W | 40–100°C/W |

If ΔT < 1°C: sensor is not close enough to the chip.  
If ΔT > 10°C: something else is heating (voltage regulator? USB chip?).

---

## Calibration Output

Running `fit_thermal.py` produces:

```
calibration/results/
├── calibration_result.json     ← full fit data + correction factors
└── thermal_fit.png             ← measured vs simulated plot

config/
└── calibration_constants.py    ← auto-injected into sim_engine
```

The `calibration_constants.py` file contains correction factors that are
applied to **all future simulations** — so if your chip runs 1.15× hotter
than JEDEC spec, every thermal prediction is automatically adjusted.
