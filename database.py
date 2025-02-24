from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import Config

Base = declarative_base()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    gesture = Column(String(50))
    correct_translation = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def add_feedback_entry(gesture, correct_translation):
    session = Session()
    try:
        feedback = Feedback(gesture=gesture, correct_translation=correct_translation)
        session.add(feedback)
        session.commit()
        return True
    except Exception as e:
        print(f"Error adding feedback: {e}")
        session.rollback()
        return False
    finally:
        session.close()