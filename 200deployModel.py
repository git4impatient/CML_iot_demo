def predict(args):
  sensorsum = args["sensor1"] + args["sensor2"]+args["sensor3"]
  if (sensorsum > 2 ):
        prediction = "Device " + str(args["devid"]) + " is expected to fail "
  else:
    prediction = "Device " + str(args["devid"]) \
     + " is operating within expected limits "
  return prediction
