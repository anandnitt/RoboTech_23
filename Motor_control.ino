
void setup() {
  Serial.begin(9600);
    pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);

  //Serial1.begin(9600);
}

void loop() {
    digitalWrite(8, LOW);  // turn the LED on (HIGH is the voltage level)
  digitalWrite(9, LOW);  // turn the LED on (HIGH is the voltage level)
  digitalWrite(10, LOW);  // turn the LED on (HIGH is the voltage level)

  if (Serial.available()) {        // If anything comes in Serial (USB),
  char t = Serial.read();
  if(t == '\n')
  {}
  else if(t == 'l')
  {
  digitalWrite(8, HIGH);  // turn the LED on (HIGH is the voltage level)
//digitalWrite(9, LOW);  // turn the LED on (HIGH is the voltage level)
//  digitalWrite(10, LOW);
delay(500);    
  }
  else if (t == 'r')
  {
      digitalWrite(9, HIGH);  // turn the LED on (HIGH is the voltage level)
 //digitalWrite(8, LOW);  // turn the LED on (HIGH is the voltage level)
 // digitalWrite(10, LOW);
  delay(500);
  }
  else
  {
      digitalWrite(10, HIGH);  // turn the LED on (HIGH is the voltage level)
  //digitalWrite(9, LOW);  // turn the LED on (HIGH is the voltage level)
  //digitalWrite(8, LOW);
  delay(500);
  }
    //Serial1.write(Serial.read());  // read it and send it out Serial1 (pins 0 & 1)
  }

  //if (Serial1.available()) {       // If anything comes in Serial1 (pins 0 & 1)
  //  Serial.write(Serial1.read());  // read it and send it out Serial (USB)
  //}
}
