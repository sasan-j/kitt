# KITT: Knowledge-based Intelligence for Transportation Technologies

Presented at [IEEE VNC 2024](https://ieee-vnc.org/2024/) (Vehicle Networking Conference) as a demo titled "Demo: Towards a Conversational LLM-Based Voice Assistant for Transportation Applications"

## Abstract

Conversational assistants based on large language models (LLMs) have spread widely across many domains, and the automotive industry is keen to follow suit.  
However, current LLMs lack sufficient understanding of geospatial data; in addition, timely information, such as weather and traffic conditions, is inaccessible to LLMs.
In this demo, we present an in-car assistant capable of verbally communicating with the driver, and by utilizing external APIs, it can answer questions related to routing, finding points of interest, and is aware of the local weather and traffic conditions.
The assistant, including a customizable speech synthesizer, is accessible through a graphical user interface that facilitates experimentation by simulating the change in time, origin, destination, and location of the car.


## Description

This project integrates speech-to-text and text-to-speech functionalities into a car's infotainment system, using the LLaMA 3 model to process and respond to vocal queries from users. It employs Gradio for user interface creation, NexusRaven for function calling, and integrates various APIs to fetch real-time information, making it a comprehensive solution for creating a responsive and interactive car assistant.

## Features

•	Speech-to-Text and Text-to-Speech: Enables the car assistant to listen to spoken questions and respond audibly, providing a hands-free experience for drivers and passengers.  
•	Intelligent Function Calling with NexusRaven: Implements a sophisticated system for executing commands and retrieving information based on user queries, using the LLaMA 3 model's capabilities.  
•	Dynamic Model Integration: Incorporates multiple models for language recognition, speech processing, and text generation.  
•	User-Friendly Gradio Interface: easy-to-use interface for testing and deploying the speaking assistant within the car's infotainment system.  
•	Real-Time Information Retrieval: Capable of integrating with various APIs to provide up-to-date information on weather, routes, points of interest, and more.


## Authors

Sasan Jafarnejad
Abigail Berthe--Pardo


## License

KITT is released under the [MIT License](https://opensource.org/licenses/MIT).