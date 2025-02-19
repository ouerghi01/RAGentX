'use client';

import Header from "./Components/header";
import { useState } from "react";
import Body, { IBodyState } from "./Components/body";
import Footer from "./Components/footer";



export default function Home() {
  const [messages, setMessages] = useState<IBodyState>({ messages: [] });
  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <Header />
      <Body messages={messages.messages} />
      <Footer setMessage={setMessages} />

    </div>
  );
}
