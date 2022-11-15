import py_qmc5883l
import time

sensor = py_qmc5883l.QMC5883L()
sensor.calibration = [[1.03947, -0.03999, -950.56431],
                  [-0.03999, 1.04052, 2116.21256],
                  [0.0, 0.0, 1.0]]
                  

while True:
    sensor.declination = -130
    m = sensor.get_bearing()
    print(m)
    time.sleep(0.2)
