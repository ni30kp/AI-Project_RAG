"""Script to create a sample PDF for testing the RAG functionality."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def create_sample_pdf():
    """Create a sample PDF document about AI and Machine Learning."""
    
    doc = SimpleDocTemplate(
        "sample_document.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20
    )
    
    content = []
    
    # Title
    content.append(Paragraph("Introduction to Artificial Intelligence", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Section 1
    content.append(Paragraph("What is Artificial Intelligence?", heading_style))
    content.append(Paragraph("""
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent 
    machines that can perform tasks that typically require human intelligence. These tasks include 
    learning, reasoning, problem-solving, perception, and language understanding. AI systems can 
    be categorized into narrow AI, which is designed for specific tasks, and general AI, which 
    would have human-like cognitive abilities across all domains.
    """, styles['Normal']))
    content.append(Spacer(1, 0.15*inch))
    
    # Section 2
    content.append(Paragraph("Machine Learning Fundamentals", heading_style))
    content.append(Paragraph("""
    Machine Learning (ML) is a subset of AI that enables systems to learn and improve from 
    experience without being explicitly programmed. The key types of machine learning include:
    
    1. Supervised Learning: The algorithm learns from labeled training data to make predictions.
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
    3. Reinforcement Learning: The algorithm learns through trial and error with rewards.
    
    Popular machine learning algorithms include linear regression, decision trees, random forests,
    support vector machines, and neural networks.
    """, styles['Normal']))
    content.append(Spacer(1, 0.15*inch))
    
    # Section 3
    content.append(Paragraph("Deep Learning and Neural Networks", heading_style))
    content.append(Paragraph("""
    Deep Learning is a specialized form of machine learning that uses artificial neural networks 
    with multiple layers (hence "deep"). These networks are inspired by the structure of the 
    human brain. Key architectures include:
    
    - Convolutional Neural Networks (CNNs): Excellent for image recognition and computer vision.
    - Recurrent Neural Networks (RNNs): Designed for sequential data like text and time series.
    - Transformers: The foundation of modern language models like GPT and BERT.
    
    Deep learning has achieved remarkable success in areas such as image classification, natural 
    language processing, speech recognition, and game playing.
    """, styles['Normal']))
    content.append(Spacer(1, 0.15*inch))
    
    # Section 4
    content.append(Paragraph("Large Language Models (LLMs)", heading_style))
    content.append(Paragraph("""
    Large Language Models are AI systems trained on massive amounts of text data. They can 
    understand and generate human-like text. Notable examples include:
    
    - GPT (Generative Pre-trained Transformer): Developed by OpenAI, known for text generation.
    - BERT: Developed by Google, excellent at understanding context in text.
    - LLaMA: Meta's open-source language model family.
    - Claude: Anthropic's AI assistant focused on safety and helpfulness.
    
    LLMs power applications like chatbots, content generation, code assistance, and translation.
    They use the transformer architecture with attention mechanisms to process and generate text.
    """, styles['Normal']))
    content.append(Spacer(1, 0.15*inch))
    
    # Section 5
    content.append(Paragraph("Applications of AI", heading_style))
    content.append(Paragraph("""
    AI has numerous practical applications across industries:
    
    - Healthcare: Disease diagnosis, drug discovery, personalized medicine.
    - Finance: Fraud detection, algorithmic trading, credit scoring.
    - Transportation: Autonomous vehicles, route optimization, traffic prediction.
    - Education: Personalized learning, automated grading, tutoring systems.
    - Entertainment: Recommendation systems, content creation, gaming AI.
    
    The global AI market is expected to reach $190 billion by 2025, demonstrating the 
    technology's growing importance in the modern economy.
    """, styles['Normal']))
    content.append(Spacer(1, 0.15*inch))
    
    # Section 6
    content.append(Paragraph("Ethical Considerations", heading_style))
    content.append(Paragraph("""
    As AI becomes more prevalent, ethical considerations become increasingly important:
    
    - Bias and Fairness: AI systems can perpetuate or amplify existing biases in training data.
    - Privacy: AI often requires large amounts of personal data, raising privacy concerns.
    - Job Displacement: Automation may replace certain jobs while creating new ones.
    - Accountability: Determining responsibility when AI systems make mistakes.
    - Transparency: Understanding how AI makes decisions (explainability).
    
    Responsible AI development requires careful attention to these issues and ongoing dialogue
    between technologists, policymakers, and society.
    """, styles['Normal']))
    
    doc.build(content)
    print("Created: sample_document.pdf")

if __name__ == "__main__":
    create_sample_pdf()
