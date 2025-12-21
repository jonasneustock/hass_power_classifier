(function () {
  function renderLineChart(canvas, points, events) {
    var ctx = canvas.getContext("2d");
    var width = canvas.width;
    var height = canvas.height;
    var padding = 24;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbf9f6";
    ctx.fillRect(0, 0, width, height);

    if (!points || points.length < 2) {
      ctx.fillStyle = "#6f665c";
      ctx.font = "14px 'Space Grotesk', sans-serif";
      ctx.fillText("Not enough data to draw a chart", padding, height / 2);
      return;
    }

    var values = points.map(function (p) { return Number(p.value); });
    var times = points.map(function (p) { return Number(p.ts); });
    var minVal = Math.min.apply(null, values);
    var maxVal = Math.max.apply(null, values);
    var span = maxVal - minVal;
    if (span === 0) {
      span = 1;
    }

    var minTs = Math.min.apply(null, times);
    var maxTs = Math.max.apply(null, times);
    var tsSpan = maxTs - minTs || 1;

    ctx.strokeStyle = "rgba(196, 93, 60, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.strokeStyle = "#c45d3c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach(function (point, index) {
      var x = padding + ((Number(point.ts) - minTs) / tsSpan) * (width - padding * 2);
      var normalized = (Number(point.value) - minVal) / span;
      var y = height - padding - normalized * (height - padding * 2);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    if (events && events.length) {
      events.forEach(function (ev) {
        var ex = padding + ((Number(ev.ts) - minTs) / tsSpan) * (width - padding * 2);
        ctx.strokeStyle = ev.phase === "start" ? "#2f855a" : "#9b2c2c";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(ex, padding);
        ctx.lineTo(ex, height - padding);
        ctx.stroke();
        if (canvas.dataset.showLabels === "true") {
          ctx.fillStyle = ctx.strokeStyle;
          ctx.font = "11px 'Space Grotesk', sans-serif";
          ctx.fillText(ev.appliance || ev.phase, ex + 4, padding + 12);
        }
      });
    }

    ctx.fillStyle = "#6f665c";
    ctx.font = "12px 'Space Grotesk', sans-serif";
    ctx.fillText(minVal.toFixed(1) + " W", padding, height - 8);
    ctx.fillText(maxVal.toFixed(1) + " W", padding, padding - 6);
  }

  function initCharts() {
    var charts = document.querySelectorAll("canvas[data-chart]");
    charts.forEach(function (canvas) {
      try {
        var points = JSON.parse(canvas.dataset.chart);
        var events = [];
        if (canvas.dataset.events) {
          events = JSON.parse(canvas.dataset.events);
        }
        renderLineChart(canvas, points, events);
      } catch (error) {
        console.warn("Failed to render chart", error);
      }
    });

    var groupCharts = document.querySelectorAll("canvas[data-chart-group]");
    groupCharts.forEach(function (canvas) {
      try {
        var groupData = JSON.parse(canvas.dataset.chartGroup || "{}");
        var combined = [];
        Object.keys(groupData).forEach(function (sensor) {
          var series = groupData[sensor] || [];
          series.forEach(function (p) {
            combined.push({ ts: p.ts, value: p.value, sensor: sensor });
          });
        });
        combined.sort(function (a, b) {
          return Number(a.ts) - Number(b.ts);
        });
        var events = [];
        if (canvas.dataset.events) {
          events = JSON.parse(canvas.dataset.events);
        }
        renderLineChart(canvas, combined, events);
      } catch (error) {
        console.warn("Failed to render grouped chart", error);
      }
    });
  }

  async function refreshLogs() {
    var feeds = document.querySelectorAll("[data-log-feed]");
    if (!feeds.length) return;
    try {
      const res = await fetch("/logs/feed");
      if (!res.ok) return;
      const data = await res.json();
      const logs = data.logs || [];
      feeds.forEach(function (feed) {
        feed.innerHTML = logs
          .slice(0, 30)
          .map(function (log) {
            return (
              '<div class="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 mb-2">' +
              '<p class="font-semibold text-slate-800">' +
              log.message +
              "</p>" +
              '<p class="text-xs text-slate-500">' +
              log.ts +
              " • " +
              log.level +
              "</p>" +
              "</div>"
            );
          })
          .join("");
      });
    } catch (e) {
      console.warn("Failed to refresh logs", e);
    }
  }

  function initLogs() {
    refreshLogs();
    setInterval(refreshLogs, 10000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initCharts);
    document.addEventListener("DOMContentLoaded", initLogs);
  } else {
    initCharts();
    initLogs();
  }
})();
