String incomingByte ;    
int pin= 4;
void setup() {

  Serial.begin(115200);

  pinMode(pin, OUTPUT);

}
void loop() {

  if (Serial.available()) {

  incomingByte = Serial.readStringUntil('\n');
  float distance = incomingByte.toFloat();
    if(distance>=100.0)
    {
      digitalWrite(pin, HIGH);
      Serial.println("ON");
    }
    else
    {
      digitalWrite(pin, LOW);
      Serial.println("OFF");
    }
  }
  else
  {
    digitalWrite(pin, HIGH);  // turn the LED on (HIGH is the voltage level)
    delay(1000);                      // wait for a second
    digitalWrite(pin, LOW);   // turn the LED off by making the voltage LOW
    delay(1000); 
  }

  }

