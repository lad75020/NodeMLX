import { MongoClient, ObjectId } from "mongodb";

export async function createChatStore({ url, databaseName }) {
  const { client, collection } = await connectMongo({ url, databaseName });

  const createChat = async (userId) => {
    if (!collection) throw new Error("Chat storage unavailable.");
    if (typeof userId !== "number") throw new Error("Missing user id.");
    const now = new Date();
    const result = await collection.insertOne({
      userId,
      startedAt: now,
      title: null,
      messages: [],
    });
    return {
      id: result.insertedId.toString(),
      startedAt: now.toISOString(),
      title: null,
      messageCount: 0,
    };
  };

  const appendChatMessages = async (userId, chatId, entries) => {
    if (!collection || !chatId || typeof userId !== "number") return;
    let id;
    try {
      id = new ObjectId(chatId);
    } catch {
      return;
    }
    const update = { $push: { messages: { $each: entries } } };
    const firstUserEntry = entries.find(
      (entry) => entry.role === "user" && entry.text,
    );
    if (firstUserEntry) {
      const document = await collection.findOne(
        { _id: id, userId },
        { projection: { title: 1 } },
      );
      if (document && !document.title)
        update.$set = { title: firstUserEntry.text.slice(0, 80) };
    }
    await collection.updateOne({ _id: id, userId }, update);
  };

  const ensureUserChat = async (userId, chatId) => {
    if (!collection) return { chatId: null, created: null };
    if (chatId) {
      try {
        const id = new ObjectId(chatId);
        const exists = await collection.findOne(
          { _id: id, userId },
          { projection: { _id: 1 } },
        );
        if (exists) return { chatId, created: null };
      } catch {}
    }
    const created = await createChat(userId);
    return { chatId: created.id, created };
  };

  return { client, collection, createChat, appendChatMessages, ensureUserChat };
}

export async function connectMongo({ url, databaseName }) {
  const client = new MongoClient(url);
  let collection = null;
  try {
    await client.connect();
    collection = client.db(databaseName).collection("Chats");
    await collection.createIndex({ startedAt: -1 });
    await collection.createIndex({ userId: 1, startedAt: -1 });
    console.log(`Mongo → ${url}/${databaseName}`);
  } catch (err) {
    console.error("MongoDB connection failed:", err.message);
  }
  return { client, collection };
}
