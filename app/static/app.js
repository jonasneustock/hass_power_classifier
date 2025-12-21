(function () {
  function renderLineChart(canvas, points) {
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
    var minVal = Math.min.apply(null, values);
    var maxVal = Math.max.apply(null, values);
    var span = maxVal - minVal;
    if (span === 0) {
      span = 1;
    }

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
      var x = padding + (index / (points.length - 1)) * (width - padding * 2);
      var normalized = (Number(point.value) - minVal) / span;
      var y = height - padding - normalized * (height - padding * 2);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

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
        renderLineChart(canvas, points);
      } catch (error) {
        console.warn("Failed to render chart", error);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initCharts);
  } else {
    initCharts();
  }
})();
