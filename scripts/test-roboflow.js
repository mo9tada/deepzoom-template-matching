const fs = require("fs");
const https = require("https");

const apiKey = process.env.ROBOFLOW_API_KEY;
if (!apiKey) {
  console.error("ROBOFLOW_API_KEY is not set");
  process.exit(1);
}

const baseUrl = (process.env.ROBOFLOW_API_URL ?? "https://detect.roboflow.com").replace(/\/$/, "");

const resolvePath = () => {
  const workflowId = process.env.ROBOFLOW_WORKFLOW_ID?.trim();
  if (workflowId) {
    return `workflow/${workflowId.replace(/^workflow\//, "")}`;
  }
  const modelId = process.env.ROBOFLOW_MODEL_ID?.trim();
  if (modelId) {
    const version = process.env.ROBOFLOW_MODEL_VERSION?.trim();
    if (version) {
      return `${modelId.replace(/\/$/, "")}/${version}`;
    }
    return modelId;
  }
  return "floor-plan-ai-object-detection-wi6i0/1";
};

const url = `${baseUrl}/${resolvePath()}?api_key=${apiKey}&format=json`;
const imagePath = process.env.ROBOFLOW_TEST_IMAGE ?? "C:/Users/mokta/Pictures/Screenshots/test.png";
const data = fs.readFileSync(imagePath);

const req = https.request(
  url,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Length": data.length,
    },
  },
  (res) => {
    let body = "";
    res.on("data", (chunk) => (body += chunk));
    res.on("end", () => {
      console.log("status", res.statusCode);
      console.log(body);
    });
  }
);

req.on("error", (err) => {
  console.error(err);
  process.exit(1);
});

req.write(data);
req.end();