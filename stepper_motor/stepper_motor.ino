const int stepPin = 3; 
const int dirPin = 2; 
const int sleepPin = 4; // Tying this to the TMC2209 EN (Enable) pin
const int potPin = A0;

// --- Physical Parameters (NEMA 17 Rotary) ---
const float AMPLITUDE_RAD = 1.0;          // Max rotation in radians (1.0 rad ~ 57.3 deg)
const float freq_min = 0.1;               // Lowest frequency (Hz)
const float freq_max = 15.0;              // Upper limit tailored to NEMA 17 torque

// --- Hardware Conversions ---
const int MICROSTEPS = 4;                 // Set to 4 for 1/4 microstepping
const float STEPS_PER_REV = 200.0 * MICROSTEPS; // 800 steps/rev
const float STEPS_PER_RAD = STEPS_PER_REV / TWO_PI;
const int AMPLITUDE_STEPS = round(AMPLITUDE_RAD * STEPS_PER_RAD); 

// --- Precomputed Log Values ---
const float log_min = log10(freq_min);
const float log_max = log10(freq_max);

// --- State Variables ---
float current_frequency = 0.1;
float phase = 0.0;
long current_step = 0;
long target_step = 0;

// --- Timers ---
unsigned long last_kinematic_update = 0;
unsigned long last_serial_update = 0;
unsigned long last_step_time = 0;

const unsigned long KINEMATIC_INTERVAL = 1000; // 1ms = 1000 Hz update
const unsigned long SERIAL_INTERVAL = 100;     // 100ms = 10 Hz UI update
const unsigned long MIN_STEP_DELAY = 100;      // 100us minimum between steps

void setup() {
  pinMode(stepPin, OUTPUT); 
  pinMode(dirPin, OUTPUT);
  pinMode(sleepPin, OUTPUT); // Acts as our Enable control
  pinMode(potPin, INPUT);
  
  digitalWrite(sleepPin, LOW); // TMC2209 EN pin is active LOW (LOW = Motor ON)
  Serial.begin(115200);
}

void loop() {
  unsigned long current_micros = micros();
  unsigned long current_millis = millis();

  // 1. UI LOOP
  if (current_millis - last_serial_update >= SERIAL_INTERVAL) {
    last_serial_update = current_millis;
    
    int potValue = analogRead(potPin);
    float normalized = (float)potValue / 1023.0;
    
    float log_freq = log_min + normalized * (log_max - log_min);
    current_frequency = pow(10, log_freq);

    // REMINDER: Comment out during actual high-speed camera collection
    Serial.print("Freq (Hz): ");
    Serial.println(current_frequency);
  }

  // 2. KINEMATICS LOOP
  if (current_micros - last_kinematic_update >= KINEMATIC_INTERVAL) {
    last_kinematic_update += KINEMATIC_INTERVAL;

    phase += TWO_PI * current_frequency * 0.001;
    if (phase >= TWO_PI) phase -= TWO_PI;

    target_step = round(AMPLITUDE_STEPS * sin(phase));
  }

  // 3. STEPPER DRIVER LOOP
  if (current_step != target_step) {
    if (current_micros - last_step_time >= MIN_STEP_DELAY) {
      
      if (target_step > current_step) {
        digitalWrite(dirPin, HIGH);
        current_step++;
      } else {
        digitalWrite(dirPin, LOW);
        current_step--;
      }
      
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(2); 
      digitalWrite(stepPin, LOW);
      
      last_step_time = current_micros;
    }
  }
}