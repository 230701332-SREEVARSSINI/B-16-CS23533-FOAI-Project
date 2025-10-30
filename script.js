function fetchLiveData() {
  fetch('/api/live-data')
      .then(response => response.json())
      .then(data => {
          document.getElementById("tempValue").textContent = data.temperature + "°C";
          updateGauge("tempGaugeFill", data.temperature, 120);

          document.getElementById("rpmValue").textContent = data.rpm;
          updateGauge("rpmGaugeFill", data.rpm, 6000);

          if (data.engine_load != null) {
              document.getElementById("engineLoadValue").textContent = parseFloat(data.engine_load).toFixed(1) + "%";
          } else {
              console.warn("engine_load is missing from API data");
              document.getElementById("engineLoadValue").textContent = "N/A";
          }

          document.getElementById("o2SensorValue").textContent = data.o2_sensor.toFixed(2) + "V";
          document.getElementById("fuelEfficiency").textContent = data.fuel_efficiency + " km/L";
          document.getElementById("systemHealth").textContent = data.system_health + "%";
          document.getElementById("emissions").textContent = data.emissions;
          document.getElementById("maintenanceDue").textContent = data.maintenance_due;

          // Costs
          const costItemsDiv = document.getElementById("costItems");
          costItemsDiv.innerHTML = '';
          let totalFuture = 0;

          if (data.issues.length > 0) {
              document.getElementById("issuesPanel").style.display = "block";
              document.getElementById("costBreakdown").style.display = "block";

              const issue = data.issues[0];

              document.getElementById("immediateCost").textContent = `₹${issue.immediate_cost}`;

              issue.future_costs.forEach(cons => {
                  const div = document.createElement("div");
                  div.innerHTML = `
                      <strong>${cons.description}</strong><br/>
                      Failures: ${cons.future_failures.join(", ")}<br/>
                      Cost: ₹${cons.future_cost}
                  `;
                  costItemsDiv.appendChild(div);
                  totalFuture += cons.future_cost;
              });

              document.getElementById("futureCost").textContent = `₹${totalFuture}`;
          } else {
              document.getElementById("issuesPanel").style.display = "none";
              document.getElementById("costBreakdown").style.display = "none";
              document.getElementById("immediateCost").textContent = "₹0";
              document.getElementById("futureCost").textContent = "₹0";
          }

          // Alerts
          const alertsDiv = document.getElementById("systemAlerts");
          alertsDiv.innerHTML = '';
          if (data.alerts.length > 0) {
              data.alerts.forEach(alert => {
                  const div = document.createElement("div");
                  div.className = alert[0].toLowerCase(); // assuming 'CRITICAL'/'WARNING'
                  div.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${alert[1]}`;
                  alertsDiv.appendChild(div);
              });
          } else {
              alertsDiv.innerHTML = `
                  <div class="status-normal">
                      <i class="fas fa-check-circle"></i> All systems normal
                  </div>
              `;
          }

          // Timestamp
          document.getElementById("lastUpdateTime").textContent = new Date().toLocaleTimeString();
      })
      .catch(err => console.error("Error fetching data:", err));
}

function updateGauge(gaugeFillId, value, maxValue) {
  const circle = document.getElementById(gaugeFillId);
  if (!circle) return;

  const radius = circle.r.baseVal.value;
  const circumference = 2 * Math.PI * radius;
  const clampedValue = Math.max(0, Math.min(value, maxValue));
  const offset = circumference - (clampedValue / maxValue) * circumference;

  circle.style.strokeDasharray = `${circumference}`;
  circle.style.strokeDashoffset = `${offset}`;
}

// Start fetching every 1.5 seconds
setInterval(fetchLiveData, 1500);
fetchLiveData(); // Initial call
