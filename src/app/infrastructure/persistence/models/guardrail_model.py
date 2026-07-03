from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.infrastructure.persistence.models.base import Base

class Guardrail(Base):
    __tablename__ = "guardrail"

    id = Column("guardrail_id", Integer, primary_key=True, autoincrement=True)
    chat_log_id = Column(Integer, ForeignKey("chat_log.chat_log_id"), nullable=False)
    reason = Column(String, nullable=False)  # Stores the Enum string value (e.g., "BLOCKED_PROFANITY")

    # BaseEntity fields
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)

    # Relationships
    chat_log = relationship("ChatLog", backref="guardrail_logs")

    def __repr__(self):
        return f"<Guardrail(id={self.id}, chat_log_id={self.chat_log_id}, reason={self.reason})>"
