/*
 * arduino_stress_test.ino
 * 
 * Physical calibration harness for sim_engine thermal validation.
 * 
 * TARGET: ATmega328P @ 5V / 16MHz (Arduino Uno)
 * PURPOSE: Measure real R_theta and thermal time constant τ = R×C
 *          to validate and calibrate the coupled thermal solver.
 * 
 * HARDWARE CONNECTIONS (choose one temp sensor):
 * 
 *   Option A — DS18B20 (recommended, ±0.0625°C):
 *     DS18B20 DQ  → Pin 2 (with 4.7kΩ pullup to 5V)
 *     DS18B20 VCC → 5V
 *     DS18B20 GND → GND
 * 
 *   Option B — LM35 (simpler, ±0.1°C with 10-bit ADC):
 *     LM35 VOUT → A0
 *     LM35 VS   → 5V
 *     LM35 GND  → GND
 * 
 *   Option C — NTC Thermistor 10kΩ B=3950 (cheapest):
 *     Thermistor → A0 with 10kΩ voltage divider to GND
 * 
 *   IMPORTANT: Mount sensor AS CLOSE AS POSSIBLE to ATmega328P IC.
 *   On Arduino Uno: directly between pins 7-8 or under IC if using thin sensor.
 * 
 * SERIAL OUTPUT FORMAT (115200 baud):
 *   CAL,<timestamp_ms>,<phase>,<temp_c_x100>,<mode_int>,<loop_count>
 * 
 *   phase: IDLE or ACTIVE
 *   temp_c_x100: temperature × 100 (integer, e.g., 2534 = 25.34°C)
 *   mode_int: 0=idle, 1=active
 *   loop_count: number of ALU loops completed (measures actual utilization)
 * 
 * PROTOCOL:
 *   1. IDLE phase (120s): CPU sleep/idle, measure baseline temperature
 *   2. RAMP phase (signal only): sends START_ACTIVE marker
 *   3. ACTIVE phase (480s): 100% CPU stress, measure thermal rise
 *   4. COOL phase: sends END_ACTIVE, CPU returns to idle
 *   5. Repeats for second run (averaging)
 * 
 * PYTHON CALIBRATOR (sim_engine/calibration/fit_thermal.py) reads this
 * serial output and fits: T(t) = T_ss - ΔT × exp(-t/τ)
 * to extract R_theta [°C/W] and C_thermal [J/K].
 */

// ── Configuration ─────────────────────────────────────────────────────────
#define SENSOR_TYPE       1    // 1=LM35 (A0), 2=DS18B20 (pin 2), 3=NTC (A0)
#define IDLE_DURATION_S   120  // seconds of idle before stress
#define ACTIVE_DURATION_S 480  // seconds of stress test (8 min >> 3τ ≈ 163s)
#define SAMPLE_INTERVAL_S 5    // temperature sample every 5 seconds
#define SERIAL_BAUD       115200
#define LED_PIN           13   // built-in LED: ON during active phase

// Known ATmega328P power (from datasheet, verified at factory):
// These are the EXPECTED values — calibration will confirm or correct them.
#define P_ACTIVE_MW       75.0f   // mW at 5V, 16MHz, 100% utilization
#define P_IDLE_MW         27.5f   // mW in idle mode (ADC + timers running)
#define VREF_MV           1100    // internal 1.1V reference for ADC

// ── DS18B20 (optional — only if SENSOR_TYPE == 2) ─────────────────────────
#if SENSOR_TYPE == 2
  // Using OneWire library: install via Library Manager
  // Sketch > Include Library > Manage Libraries > search "DallasTemperature"
  #include <OneWire.h>
  #include <DallasTemperature.h>
  #define ONE_WIRE_BUS 2
  OneWire oneWire(ONE_WIRE_BUS);
  DallasTemperature sensors(&oneWire);
#endif

// ── Globals ────────────────────────────────────────────────────────────────
volatile uint32_t loop_count = 0;
uint32_t phase_start_ms      = 0;
bool in_active_phase         = false;
char phase_name[8]           = "IDLE";

// ── Utility: read temperature in °C × 100 (integer) ──────────────────────
int32_t read_temp_x100() {
  #if SENSOR_TYPE == 1
    // LM35: 10mV/°C, using internal 1.1V Vref for better resolution
    // With 1.1V Vref: 1 ADC step = 1.1V/1024 = 1.074mV = 0.1074°C
    analogReference(INTERNAL);
    delay(10);  // settle after reference change
    int raw = analogRead(A0);
    // LM35: Vout = T(°C) × 10mV
    // T = Vout / 10mV = (raw × Vref / 1024) / 10mV
    // T × 100 = raw × 1100 / 1024 / 10 = raw × 1100 / 10240
    return (int32_t)raw * 1100L / 10240L;

  #elif SENSOR_TYPE == 2
    // DS18B20: 12-bit resolution, ±0.0625°C steps
    sensors.requestTemperatures();
    float t = sensors.getTempCByIndex(0);
    if (t == DEVICE_DISCONNECTED_C) return -9999;
    return (int32_t)(t * 100.0f);

  #elif SENSOR_TYPE == 3
    // NTC 10kΩ B=3950K voltage divider (R_fixed = 10kΩ to GND)
    // V_ntc = VCC × R_ntc / (R_fixed + R_ntc)
    // R_ntc = R_fixed × V_ntc / (VCC - V_ntc)
    // T = 1 / (1/T0 + ln(R/R0)/B) - 273.15
    analogReference(DEFAULT);  // 5V reference for NTC
    delay(10);
    int raw = analogRead(A0);
    // Avoid div-by-zero
    if (raw <= 0 || raw >= 1023) return -9999;
    float r_ntc = 10000.0f * raw / (1023.0f - raw);
    // Steinhart-Hart simplified (B parameter model):
    // 1/T = 1/T0 + (1/B) × ln(R/R0)
    // T0 = 298.15K (25°C), R0 = 10kΩ, B = 3950K
    float log_r = log(r_ntc / 10000.0f);
    float inv_t = (1.0f / 298.15f) + (log_r / 3950.0f);
    float temp_c = (1.0f / inv_t) - 273.15f;
    return (int32_t)(temp_c * 100.0f);

  #else
    return -9999;  // Unknown sensor
  #endif
}

// ── Stress loop: pure ALU work, no memory accesses to avoid heating cache ──
// Uses: multiply-accumulate chain — predictable, consistent power draw
void __attribute__((optimize("O0"))) stress_burst(uint16_t ms) {
  uint32_t end_time = millis() + ms;
  volatile uint32_t a = 12345, b = 67890;
  while ((uint32_t)millis() < end_time) {
    // Unrolled inner loop: 20 multiply-accumulate ops per outer iteration
    // Each op = ~2 cycles at 16MHz → ~12.5ns
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    a = a * 31337 + b; b = b * 73939 + a;
    loop_count++;
    // Prevent a/b from being optimized away
    if (a == 0 && b == 0) digitalWrite(LED_PIN, HIGH);
  }
}

// ── Sample and transmit one data point ────────────────────────────────────
void transmit_sample(uint32_t elapsed_s) {
  int32_t temp = read_temp_x100();
  Serial.print("CAL,");
  Serial.print(millis());
  Serial.print(",");
  Serial.print(phase_name);
  Serial.print(",");
  Serial.print(temp);         // T × 100 (e.g., 2534 = 25.34°C)
  Serial.print(",");
  Serial.print(in_active_phase ? 1 : 0);
  Serial.print(",");
  Serial.print(loop_count);
  Serial.print(",");
  Serial.println(elapsed_s);  // seconds since phase start
}

// ── Setup ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {}
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  #if SENSOR_TYPE == 2
    sensors.begin();
    sensors.setResolution(12);  // max resolution: 0.0625°C
  #endif

  // Header
  Serial.println("# sim_engine Arduino Thermal Calibration Harness v3");
  Serial.println("# ATmega328P @ 5V/16MHz");
  Serial.print("# SENSOR_TYPE="); Serial.println(SENSOR_TYPE);
  Serial.print("# P_ACTIVE_MW="); Serial.println(P_ACTIVE_MW);
  Serial.print("# P_IDLE_MW=");   Serial.println(P_IDLE_MW);
  Serial.println("# Format: CAL,timestamp_ms,phase,temp_x100,mode_int,loop_count,elapsed_s");
  Serial.println("# START");
}

// ── Main loop ─────────────────────────────────────────────────────────────
void loop() {
  uint32_t run_start = millis();
  loop_count = 0;
  
  // ── Phase 1: IDLE ──────────────────────────────────────────────────────
  in_active_phase = false;
  strcpy(phase_name, "IDLE");
  digitalWrite(LED_PIN, LOW);
  Serial.println("# PHASE:IDLE START");
  phase_start_ms = millis();

  uint32_t idle_end = millis() + (uint32_t)IDLE_DURATION_S * 1000;
  uint32_t next_sample = millis();
  
  while ((uint32_t)millis() < idle_end) {
    if ((uint32_t)millis() >= next_sample) {
      uint32_t elapsed = ((uint32_t)millis() - phase_start_ms) / 1000;
      transmit_sample(elapsed);
      next_sample += (uint32_t)SAMPLE_INTERVAL_S * 1000;
    }
    delay(100);  // real idle — let CPU rest
  }
  Serial.println("# PHASE:IDLE END");

  // ── Phase 2: ACTIVE ────────────────────────────────────────────────────
  in_active_phase = true;
  strcpy(phase_name, "ACTIVE");
  digitalWrite(LED_PIN, HIGH);
  Serial.println("# PHASE:ACTIVE START");
  phase_start_ms = millis();
  loop_count = 0;
  next_sample = millis();

  uint32_t active_end = millis() + (uint32_t)ACTIVE_DURATION_S * 1000;

  while ((uint32_t)millis() < active_end) {
    if ((uint32_t)millis() >= next_sample) {
      uint32_t elapsed = ((uint32_t)millis() - phase_start_ms) / 1000;
      transmit_sample(elapsed);
      next_sample += (uint32_t)SAMPLE_INTERVAL_S * 1000;
    }
    // Stress: 900ms on, 100ms off (for temperature sampling window)
    stress_burst(900);
    delay(100);
  }

  Serial.println("# PHASE:ACTIVE END");
  digitalWrite(LED_PIN, LOW);

  // ── Wait before next run ───────────────────────────────────────────────
  Serial.println("# COOLING_DOWN");
  delay(300000);  // 5 min cooldown before repeat
}
