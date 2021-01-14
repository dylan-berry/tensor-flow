require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');

// Main function
const calcSimilarity = async () => {
  const model = await use.load();

  sentenceOne = `I like to swim.`;
  sentenceTwo = `I enjoy playing water polo.`;

  const embeddings = await model.embed([sentenceOne, sentenceTwo]);
  const res = await embeddings.array();
  const similarity = cosineSimilarity(res[0], res[1]);
  console.log(`${(similarity * 100).toFixed(2)}% similarity`);
};

calcSimilarity();

// Helper functions
const dotProduct = (a, b) => {
  let product = 0;
  for (let i = 0; i < a.length; i++) {
    product += a[i] * b[i];
  }
  return product;
};

const magnitude = vector => {
  let sum = 0;
  for (let value of vector) {
    sum += value * value;
  }
  return Math.sqrt(sum);
};

const cosineSimilarity = (a, b) => {
  return dotProduct(a, b) / (magnitude(a) * magnitude(b));
};
